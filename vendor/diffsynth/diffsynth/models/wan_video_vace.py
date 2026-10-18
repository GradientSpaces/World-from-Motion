import torch
from .wan_video_dit import DiTBlock
from ..core.gradient import gradient_checkpoint_forward


_TARGET_GBUFFER_DROPOUT_SCOPES = ("target_only", "full_context")
_OPTIONAL_STATE_DICT_KEYS = frozenset(
    {
        "target_only_uncond_vace_context",
        "full_context_uncond_vace_context",
    }
)

class VaceWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs, cam_emb=None):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, context, t_mod, freqs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class VaceWanModel(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=96,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # vace blocks
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = torch.nn.Conv3d(vace_in_dim, dim, kernel_size=patch_size, stride=patch_size)
        # Register the learned-unconditional VACE contexts as BUFFERS by default.
        # Buffers still appear in state_dict (for checkpoint compatibility) but
        # are NOT iterated by self.parameters(), so:
        #   - they never enter the optimizer's flat buffer (no DeepSpeed Zero
        #     optimizer-state size mismatch on resume),
        #   - they never hit DeepSpeed's frozen-param restore path (no KeyError
        #     for missing entries in saved_frozen_params).
        # When a training run actually needs to learn them
        # (target_gbuffer_uncond_fill_mode == "learned"), the caller must
        # promote them to nn.Parameter via promote_uncond_vace_contexts_to_parameters().
        self.register_buffer(
            "target_only_uncond_vace_context",
            self._build_uncond_vace_context_tensor(vace_in_dim),
        )
        self.register_buffer(
            "full_context_uncond_vace_context",
            self._build_uncond_vace_context_tensor(vace_in_dim),
        )

    @staticmethod
    def _build_uncond_vace_context_tensor(
        vace_in_dim: int,
        init_value: float = 0.01,
        *,
        device=None,
        dtype=None,
    ):
        return torch.full(
            (1, int(vace_in_dim), 1, 1, 1),
            float(init_value),
            device=device,
            dtype=dtype,
        )

    def _get_materialized_reference_tensor(self):
        for tensor in (
            getattr(self.vace_patch_embedding, "weight", None),
            getattr(self, "target_only_uncond_vace_context", None),
            getattr(self, "full_context_uncond_vace_context", None),
        ):
            if isinstance(tensor, torch.Tensor) and not tensor.is_meta:
                return tensor
        return None

    def _replace_uncond_tensor(self, name: str, tensor: torch.Tensor):
        # Replace an existing Parameter/buffer with the given tensor, preserving
        # whichever kind it currently is. This keeps the module structure stable
        # across reinitialization/materialization.
        was_parameter = name in getattr(self, "_parameters", {})
        if was_parameter:
            # pop from parameters so setattr does not treat as buffer
            existing = self._parameters.pop(name)
            requires_grad = existing.requires_grad if existing is not None else False
            new_param = torch.nn.Parameter(tensor, requires_grad=requires_grad)
            self.register_parameter(name, new_param)
            return
        # Otherwise treat as buffer (including fresh registration).
        if name in getattr(self, "_buffers", {}):
            del self._buffers[name]
        self.register_buffer(name, tensor)

    def reset_uncond_vace_contexts(self, vace_in_dim: int | None = None, init_value: float = 0.01):
        if vace_in_dim is None:
            vace_in_dim = self.vace_in_dim
        vace_in_dim = int(vace_in_dim)
        reference_tensor = self._get_materialized_reference_tensor()
        tensor_kwargs = {}
        if reference_tensor is not None:
            tensor_kwargs["device"] = reference_tensor.device
            tensor_kwargs["dtype"] = reference_tensor.dtype
        for name in _OPTIONAL_STATE_DICT_KEYS:
            self._replace_uncond_tensor(
                name,
                self._build_uncond_vace_context_tensor(
                    vace_in_dim,
                    init_value=init_value,
                    **tensor_kwargs,
                ),
            )

    def _materialize_missing_optional_state(self, missing_keys, init_value: float = 0.01):
        missing_optional_keys = [
            key for key in missing_keys
            if key in _OPTIONAL_STATE_DICT_KEYS
        ]
        if not missing_optional_keys:
            return
        reference_tensor = self._get_materialized_reference_tensor()
        tensor_kwargs = {}
        if reference_tensor is not None:
            tensor_kwargs["device"] = reference_tensor.device
            tensor_kwargs["dtype"] = reference_tensor.dtype
        for key in missing_optional_keys:
            existing = getattr(self, key, None)
            if not isinstance(existing, torch.Tensor) or not existing.is_meta:
                continue
            vace_in_dim = int(existing.shape[1]) if existing.ndim >= 2 else int(self.vace_in_dim)
            self._replace_uncond_tensor(
                key,
                self._build_uncond_vace_context_tensor(
                    vace_in_dim,
                    init_value=init_value,
                    **tensor_kwargs,
                ),
            )

    def promote_uncond_vace_contexts_to_parameters(self, init_value: float | None = None):
        """Convert the two uncond context tensors from buffers to nn.Parameters.

        Only call this when the training run actually learns them
        (target_gbuffer_uncond_fill_mode == "learned"). Must be invoked before
        the optimizer is constructed (i.e., before accelerator.prepare).
        """
        for name in _OPTIONAL_STATE_DICT_KEYS:
            existing = getattr(self, name, None)
            if existing is None:
                # Nothing to promote; build a fresh Parameter.
                reference_tensor = self._get_materialized_reference_tensor()
                kwargs = {}
                if reference_tensor is not None:
                    kwargs["device"] = reference_tensor.device
                    kwargs["dtype"] = reference_tensor.dtype
                tensor = self._build_uncond_vace_context_tensor(
                    self.vace_in_dim,
                    init_value=0.01 if init_value is None else init_value,
                    **kwargs,
                )
            elif not isinstance(existing, torch.Tensor):
                raise TypeError(
                    f"Expected tensor for {name}, got {type(existing).__name__}."
                )
            else:
                tensor = existing.detach().clone()
                if init_value is not None:
                    tensor = tensor.fill_(float(init_value))
            if name in getattr(self, "_parameters", {}):
                # Already a Parameter; ensure it tracks gradients.
                self._parameters[name].requires_grad_(True)
                continue
            if name in getattr(self, "_buffers", {}):
                del self._buffers[name]
            self.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=True))

    def get_uncond_vace_context(self, scope: str, dtype=None, device=None):
        scope = str(scope).strip().lower()
        if scope == "target_only":
            latent = self.target_only_uncond_vace_context
        elif scope == "full_context":
            latent = self.full_context_uncond_vace_context
        else:
            raise ValueError(
                f"Unsupported target-gbuffer dropout scope '{scope}'. "
                f"Expected one of: {', '.join(_TARGET_GBUFFER_DROPOUT_SCOPES)}."
            )
        if device is not None or dtype is not None:
            latent = latent.to(
                device=latent.device if device is None else device,
                dtype=latent.dtype if dtype is None else dtype,
            )
        return latent

    @classmethod
    def _uncond_vace_context_is_default_tensor(
        cls,
        tensor: torch.Tensor,
        init_value: float = 0.01,
    ) -> bool:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.is_meta
            or tensor.ndim != 5
            or tensor.shape[0] != 1
            or tuple(tensor.shape[2:]) != (1, 1, 1)
        ):
            return False
        reference = cls._build_uncond_vace_context_tensor(
            int(tensor.shape[1]),
            init_value=init_value,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return torch.equal(tensor.detach(), reference)

    def infer_preferred_uncond_vace_context_scope(
        self,
        init_value: float = 0.01,
    ) -> str | None:
        # Prefer the full-context latent when both tensors differ from the
        # initialization value, since it covers the target-only case too.
        for scope, name in (
            ("full_context", "full_context_uncond_vace_context"),
            ("target_only", "target_only_uncond_vace_context"),
        ):
            tensor = getattr(self, name, None)
            if not isinstance(tensor, torch.Tensor) or tensor.is_meta:
                continue
            if not self._uncond_vace_context_is_default_tensor(
                tensor,
                init_value=init_value,
            ):
                return scope
        return None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Older pretrained VACE checkpoints do not contain the learned
        # unconditional context tensors introduced for post-training. Allow
        # those two keys to be absent while preserving strict loading for all
        # other parameters/buffers.
        incompatible = super().load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )
        all_missing_keys = list(incompatible.missing_keys)
        self._materialize_missing_optional_state(all_missing_keys)
        missing_keys = [
            key for key in all_missing_keys
            if key not in _OPTIONAL_STATE_DICT_KEYS
        ]
        unexpected_keys = list(incompatible.unexpected_keys)
        if strict and (missing_keys or unexpected_keys):
            error_msgs = []
            if missing_keys:
                quoted_missing = ", ".join(f'"{key}"' for key in missing_keys)
                error_msgs.append(
                    f"Missing key(s) in state_dict: {quoted_missing}. "
                )
            if unexpected_keys:
                quoted_unexpected = ", ".join(
                    f'"{key}"' for key in unexpected_keys
                )
                error_msgs.append(
                    f"Unexpected key(s) in state_dict: {quoted_unexpected}. "
                )
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t"
                + "\n\t".join(error_msgs)
            )
        return type(incompatible)(missing_keys, unexpected_keys)

    def forward(
        self, x, vace_context, context, t_mod, freqs,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])
        
        for block in self.vace_blocks:
            c = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                c, x, context, t_mod, freqs
            )
            
        hints = torch.unbind(c)[:-1]
        return hints
