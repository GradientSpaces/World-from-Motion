"""Single-thread per-camera LPIPS graphs with independent activation storage."""
import threading

import torch


class PerViewLPIPS:
    """Call after_backward only after the complete fitting loss backward returns.

    Independent graph pools permit the caller's original per-camera loss order.
    A second use of a camera before that backward uses the original operations,
    so it cannot overwrite activations belonging to the earlier call.
    """

    def __init__(self, model, reference_loss):
        self.model = model
        self.reference_loss = reference_loss
        self.owner = threading.get_ident()
        self.slots = {}
        self.used = set()
        self.captures = 0
        self.replays = 0
        self.fallbacks = 0

    def after_backward(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("Reset graph buffers on their owning fitting thread")
        self.used.clear()

    def loss(self, camera, prediction, target, mask):
        def original():
            self.fallbacks += 1
            return self.reference_loss(target, {"rgb": prediction}, mask, self.model)[0]

        if (threading.get_ident() != self.owner or not torch.is_grad_enabled() or
                prediction.device.type != "cuda" or not prediction.requires_grad or
                mask.requires_grad or self.model.training or
                any(parameter.requires_grad for parameter in self.model.parameters()) or
                camera in self.used):
            return original()
        target = target.detach()
        inputs = (prediction, target, mask)
        if any(value.device != prediction.device for value in inputs):
            return original()
        stream = torch.cuda.current_stream(prediction.device).cuda_stream
        signature = (stream, tuple((tuple(value.shape), tuple(value.stride()),
                                    str(value.dtype), str(value.device)) for value in inputs))
        slot = self.slots.get(camera)
        if slot is None or slot[0] != signature:
            # Capturing detached copies cannot consume the caller's render graph.
            samples = (prediction.detach().clone().requires_grad_(), target.clone(), mask.detach().clone())

            def reference(x, y, weights):
                return self.reference_loss(y, {"rgb": x}, weights, self.model)[0]

            # One call per capture gives this camera its own forward/backward pool.
            function = torch.cuda.make_graphed_callables(reference, samples, num_warmup_iters=5)
            slot = (signature, function)
            self.slots[camera] = slot
            self.captures += 1
        self.used.add(camera)
        self.replays += 1
        return slot[1](*inputs)
