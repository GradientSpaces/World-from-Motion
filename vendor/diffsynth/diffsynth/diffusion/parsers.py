import argparse


def add_dataset_base_config(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--eval_dataset_base_path", type=str, default=None, help="Optional base path for a separate evaluation dataset.")
    parser.add_argument("--eval_dataset_metadata_path", type=str, default=None, help="Optional metadata path for a separate evaluation dataset.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    return parser

def add_image_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution.")
    return parser

def add_video_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    return parser

def add_model_config(parser: argparse.ArgumentParser):
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--fp8_models", default=None, help="Models with FP8 precision, comma-separated.")
    parser.add_argument("--offload_models", default=None, help="Models with offload, comma-separated. Only used in splited training.")
    parser.add_argument("--remove_cached_keys", default=None, help="Comma-separated keys to drop from cached inputs before training.")
    return parser

def add_training_config(parser: argparse.ArgumentParser):
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--dit_learning_rate",
        type=float,
        default=None,
        help="Optional WAN split LR for trainable DiT parameters. Must be used with --other_learning_rate.",
    )
    parser.add_argument(
        "--other_learning_rate",
        type=float,
        default=None,
        help="Optional WAN split LR for non-DiT or add-on trainable parameters. Must be used with --dit_learning_rate.",
    )
    parser.add_argument(
        "--uncond_latent_learning_rate",
        type=float,
        default=None,
        help=(
            "Optional WAN LR for the learned unconditional VACE latent parameters "
            "(*_uncond_vace_context). Only used when --dit_learning_rate is set. "
            "If unset, these parameters fall into the 'other' group."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=(
            "Optional cap on optimizer steps for the current run. "
            "When --initial_global_step is set, training stops once global_step "
            "reaches initial_global_step + max_train_steps."
        ),
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU process during training.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    return parser

def add_output_config(parser: argparse.ArgumentParser):
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    return parser

def add_lora_config(parser: argparse.ArgumentParser):
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--preset_lora_path", type=str, default=None, help="Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model.")
    parser.add_argument("--preset_lora_model", type=str, default=None, help="Which model the preset LoRA is fused to.")
    return parser

def add_gradient_config(parser: argparse.ArgumentParser):
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    return parser

def add_validation_config(parser: argparse.ArgumentParser):
    parser.add_argument("--wandb", default=False, action="store_true", help="Whether to enable wandb logging.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity name (default: your W&B default entity).")
    parser.add_argument("--wandb_project", type=str, default="world-from-motion", help="WandB project name.")
    parser.add_argument("--job_name", type=str, default=None, help="Job name for wandb run ID and output subfolder.")
    parser.add_argument("--ckpt_save_every", type=int, default=1000, help="Checkpoint saving frequency in steps.")
    parser.add_argument("--visualize_every", type=int, default=5000, help="Video logging frequency in steps.")
    parser.add_argument("--val_size", type=int, default=4, help="Maximum number of validation videos to log.")
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=True, help="Whether to auto-resume from latest checkpoint.")
    parser.add_argument("--autoresume_check_interval", type=int, default=1, help="How often (in optimizer steps) to check for termination signals and trigger checkpoint+exit.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of LR warmup steps.")
    parser.add_argument("--profile_step_time", default=False, action="store_true", help="Enable training step time profiling (data wait/forward/backward/optimizer/overhead).")
    parser.add_argument("--profile_log_interval", type=int, default=20, help="Log timing profile every N optimizer steps when --profile_step_time is enabled.")
    parser.add_argument("--profile_warmup_steps", type=int, default=5, help="Skip first N optimizer steps in timing profile aggregation.")
    parser.add_argument("--profile_output_name", type=str, default="time_profile_summary.json", help="Output filename (under output_path) for final timing profile summary JSON.")
    parser.add_argument("--eval_num_inference_steps", type=int, default=20, help="Number of denoising steps used for validation/evaluation renders.")
    return parser

def add_general_config(parser: argparse.ArgumentParser):
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_validation_config(parser)
    return parser
