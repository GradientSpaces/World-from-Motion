#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Render a camera-UI trajectory to WorldTree/DiffSynth g-buffers, then run
DiffSynth inference on the rendered camera.

Inputs:
  saved-dir: a MoSca photometric log dir, e.g. data/wild_workspaces/<scene>/logs/<run>
  camera:    latest .npz under <saved-dir>/camera_ui_exports/ unless --camera-npz is given

Usage:
  bash tools/run_camera_ui_pipeline.sh \
    --camera-npz data/wild_workspaces/<scene>/logs/<run>/camera_ui_exports/cam_ui_c2w.npz \
    --diffsynth-root /path/to/diffsynth --ckpt /path/to/checkpoint.safetensors

Options:
  --scene NAME                    Default: animal_motion_00
  --ws PATH                       Workspace. Default: data/wild_workspaces/$SCENE
  --saved-dir PATH                Photometric log dir. Inferred from --camera-npz when possible.
  --camera-npz PATH               UI trajectory. Default: newest camera_ui_exports/*.npz
  --camera-name NAME              Target camera name. Default: cam_ui
  --cfg PATH                      MoSca fit config. Default: infer --cfg or --fit_cfg from saved dir.
  --benchmark-root PATH           G-buffer output root.
  --inference-root PATH           DiffSynth output root.
  --diffsynth-root PATH           DiffSynth root.
  --ckpt PATH                     DiffSynth checkpoint.
  --frame-stride N                Default: 2
  --max-frames N                  Default: 73
  --chunk-size N                  Default: 73
  --inference-res HxW             Default: 384x512
  --pred-subdir NAME              Default: pred_rgb_first73_stride2
  --num-inference-steps N         Default: 20
  --cfg-scale FLOAT               Default: 1.0
  --target-gbuffer-cfg-scale F    Default: 1.0
  --target-gbuffer-dropout-scope  Default: target_only
  --target-gbuffer-uncond-fill-mode
                                  Default: zero
  --vace-scale FLOAT              Default: 1.0
  --seed N                        Default: 0
  --prompt TEXT                   Default: a single space
  --model-id-with-origin-paths S  Default: Wan2.1-VACE-14B paths
  --device DEVICE                 Default: cuda
  --no-scene-flow                 Do not pass --use-scene-flow to inference
  --no-split-cam-embed            Do not pass --split-cam-embed to inference
  --no-overwrite-render           Do not overwrite existing g-buffer scene root
  --skip-render                   Only run inference, assuming g-buffers exist
  --skip-inference                Only render g-buffers
  --skip-video-grid               Do not create the 1x3 comparison video
  --video-fps N                   Comparison video FPS. Default: 15
  --video-output PATH             Optional comparison video output path
  --dry-run                       Print commands without executing
  -h, --help                      Show this help
EOF
}

SCENE="animal_motion_00"
WS=""
SAVED_DIR=""
CAMERA_NPZ=""
CAMERA_NAME="cam_ui"
CFG_PATH=""
BENCHMARK_ROOT=""
INFERENCE_ROOT=""
DIFFSYNTH_ROOT=""
CKPT=""
FRAME_STRIDE=2
MAX_FRAMES=73
CHUNK_SIZE=73
INFERENCE_RES="384x512"
PRED_SUBDIR="pred_rgb_first73_stride2"
NUM_INFERENCE_STEPS=20
CFG_SCALE=1.0
TARGET_GBUFFER_CFG_SCALE=1.0
TARGET_GBUFFER_DROPOUT_SCOPE="target_only"
TARGET_GBUFFER_UNCOND_FILL_MODE="zero"
VACE_SCALE=1.0
SEED=0
PROMPT=" "
MODEL_ID_WITH_ORIGIN_PATHS="Wan-AI/Wan2.1-VACE-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-14B:Wan2.1_VAE.pth"
DEVICE="cuda"
USE_SCENE_FLOW=1
SPLIT_CAM_EMBED=1
OVERWRITE_RENDER=1
SKIP_RENDER=0
SKIP_INFERENCE=0
SKIP_VIDEO_GRID=0
VIDEO_FPS=15
VIDEO_OUTPUT=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene) SCENE="$2"; shift 2 ;;
        --ws) WS="$2"; shift 2 ;;
        --saved-dir) SAVED_DIR="$2"; shift 2 ;;
        --camera-npz) CAMERA_NPZ="$2"; shift 2 ;;
        --camera-name) CAMERA_NAME="$2"; shift 2 ;;
        --cfg) CFG_PATH="$2"; shift 2 ;;
        --benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
        --inference-root) INFERENCE_ROOT="$2"; shift 2 ;;
        --diffsynth-root) DIFFSYNTH_ROOT="$2"; shift 2 ;;
        --ckpt) CKPT="$2"; shift 2 ;;
        --frame-stride) FRAME_STRIDE="$2"; shift 2 ;;
        --max-frames) MAX_FRAMES="$2"; shift 2 ;;
        --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
        --inference-res) INFERENCE_RES="$2"; shift 2 ;;
        --pred-subdir) PRED_SUBDIR="$2"; shift 2 ;;
        --num-inference-steps) NUM_INFERENCE_STEPS="$2"; shift 2 ;;
        --cfg-scale) CFG_SCALE="$2"; shift 2 ;;
        --target-gbuffer-cfg-scale) TARGET_GBUFFER_CFG_SCALE="$2"; shift 2 ;;
        --target-gbuffer-dropout-scope) TARGET_GBUFFER_DROPOUT_SCOPE="$2"; shift 2 ;;
        --target-gbuffer-uncond-fill-mode) TARGET_GBUFFER_UNCOND_FILL_MODE="$2"; shift 2 ;;
        --vace-scale) VACE_SCALE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --model-id-with-origin-paths) MODEL_ID_WITH_ORIGIN_PATHS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --no-scene-flow) USE_SCENE_FLOW=0; shift ;;
        --no-split-cam-embed) SPLIT_CAM_EMBED=0; shift ;;
        --no-overwrite-render) OVERWRITE_RENDER=0; shift ;;
        --skip-render) SKIP_RENDER=1; shift ;;
        --skip-inference) SKIP_INFERENCE=1; shift ;;
        --skip-video-grid) SKIP_VIDEO_GRID=1; shift ;;
        --video-fps) VIDEO_FPS="$2"; shift 2 ;;
        --video-output) VIDEO_OUTPUT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ ! -f "$CKPT" ] && [ -f "${CKPT}.safetensors" ]; then
    CKPT="${CKPT}.safetensors"
fi

if [ -z "$DIFFSYNTH_ROOT" ] && [ "$SKIP_INFERENCE" -eq 0 ]; then
    echo "Error: --diffsynth-root is required unless --skip-inference is set" >&2
    usage >&2
    exit 1
fi
if [ -z "$CKPT" ] && [ "$SKIP_INFERENCE" -eq 0 ]; then
    echo "Error: --ckpt is required unless --skip-inference is set" >&2
    usage >&2
    exit 1
fi

if [ -z "$WS" ]; then
    WS="data/wild_workspaces/${SCENE}"
fi
if [ -n "$CAMERA_NPZ" ] && [ -z "$SAVED_DIR" ]; then
    camera_dir="$(dirname "$CAMERA_NPZ")"
    if [ "$(basename "$camera_dir")" = "camera_ui_exports" ]; then
        SAVED_DIR="$(dirname "$camera_dir")"
    fi
fi
if [ -n "$SAVED_DIR" ] && [ -z "${WS:-}" ]; then
    case "$SAVED_DIR" in
        */logs/*) WS="${SAVED_DIR%%/logs/*}" ;;
    esac
fi
if [ -n "$SAVED_DIR" ]; then
    case "$SAVED_DIR" in
        */logs/*)
            inferred_ws="${SAVED_DIR%%/logs/*}"
            if [ "$WS" = "data/wild_workspaces/${SCENE}" ]; then
                WS="$inferred_ws"
                SCENE="$(basename "$WS")"
            fi
            ;;
    esac
fi
if [ -z "$SAVED_DIR" ]; then
    echo "Error: pass --saved-dir, or a --camera-npz under <saved-dir>/camera_ui_exports/" >&2
    usage >&2
    exit 1
fi
if [ -z "$CFG_PATH" ] && [ -f "$SAVED_DIR/fit_commandline_args.txt" ]; then
    CFG_PATH="$(
        python - "$SAVED_DIR/fit_commandline_args.txt" <<'PY'
import shlex
import sys

tokens = shlex.split(open(sys.argv[1], encoding="utf-8").read().strip())

def get_flag(flag: str):
    prefix = flag + "="
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith(prefix):
            return token[len(prefix):]
    return None

print(get_flag("--cfg") or get_flag("--fit_cfg") or "")
PY
    )"
fi
if [ -z "$BENCHMARK_ROOT" ]; then
    BENCHMARK_ROOT="${WS}/diffsynth_camera_ui_benchmark_first${MAX_FRAMES}_stride${FRAME_STRIDE}"
fi
if [ -z "$INFERENCE_ROOT" ]; then
    INFERENCE_ROOT="${WS}/diffsynth_camera_ui_inference_first${MAX_FRAMES}_stride${FRAME_STRIDE}_sceneflow"
fi
if [ -z "$CAMERA_NPZ" ]; then
    EXPORT_DIR="${SAVED_DIR}/camera_ui_exports"
    if [[ "$EXPORT_DIR" = /* ]]; then
        EXPORT_DIR_ABS="$EXPORT_DIR"
    else
        EXPORT_DIR_ABS="$PROJECT_ROOT/$EXPORT_DIR"
    fi
    CAMERA_NPZ="$(ls -t "$EXPORT_DIR_ABS"/*.npz 2>/dev/null | head -n 1 || true)"
    if [ -z "$CAMERA_NPZ" ]; then
        echo "Error: no camera UI .npz found under $EXPORT_DIR" >&2
        exit 1
    fi
    CAMERA_NPZ="${CAMERA_NPZ#${PROJECT_ROOT}/}"
fi

cd "$PROJECT_ROOT"

RENDER_CMD=(
    python scripts/wild/render_wild_fixed_camera_gbuffers.py
    --ws "$WS"
    --saved-dir "$SAVED_DIR"
    --output-root "$BENCHMARK_ROOT"
    --external-camera-npz "$CAMERA_NPZ"
    --external-camera-align-mode none
    --camera-names "$CAMERA_NAME"
    --frame-stride "$FRAME_STRIDE"
    --max-frames "$MAX_FRAMES"
)
if [ -n "$CFG_PATH" ]; then
    RENDER_CMD+=(--cfg "$CFG_PATH")
fi
if [ "$OVERWRITE_RENDER" -eq 1 ]; then
    RENDER_CMD+=(--overwrite)
fi

INFER_CMD=(
    python run_worldtree_inference.py
    --diffsynth-root "$DIFFSYNTH_ROOT"
    --benchmark-root "$BENCHMARK_ROOT"
    --output-root "$INFERENCE_ROOT"
    --ckpt "$CKPT"
    --scene-names "$SCENE"
    --reference-camera cam0
    --target-cameras "$CAMERA_NAME"
    --chunk-size "$CHUNK_SIZE"
    --max-frames "$MAX_FRAMES"
    --pred-subdir "$PRED_SUBDIR"
    --prompt "$PROMPT"
    --inference-res "$INFERENCE_RES"
    --num-inference-steps "$NUM_INFERENCE_STEPS"
    --cfg-scale "$CFG_SCALE"
    --target-gbuffer-cfg-scale "$TARGET_GBUFFER_CFG_SCALE"
    --target-gbuffer-dropout-scope "$TARGET_GBUFFER_DROPOUT_SCOPE"
    --target-gbuffer-uncond-fill-mode "$TARGET_GBUFFER_UNCOND_FILL_MODE"
    --vace-scale "$VACE_SCALE"
    --seed "$SEED"
    --device "$DEVICE"
    --model-id-with-origin-paths "$MODEL_ID_WITH_ORIGIN_PATHS"
)
if [ "$USE_SCENE_FLOW" -eq 1 ]; then
    INFER_CMD+=(--use-scene-flow)
fi
if [ "$SPLIT_CAM_EMBED" -eq 1 ]; then
    INFER_CMD+=(--split-cam-embed)
fi

VIDEO_CMD=(
    python tools/make_camera_ui_video_grid.py
    --benchmark-root "$BENCHMARK_ROOT"
    --inference-root "$INFERENCE_ROOT"
    --scene-name "$SCENE"
    --camera-name "$CAMERA_NAME"
    --reference-camera cam0
    --pred-subdir "$PRED_SUBDIR"
    --fps "$VIDEO_FPS"
)
if [ -n "$VIDEO_OUTPUT" ]; then
    VIDEO_CMD+=(--output "$VIDEO_OUTPUT")
fi

echo "Scene:          $SCENE"
echo "Workspace:      $WS"
echo "Saved dir:      $SAVED_DIR"
echo "Camera npz:     $CAMERA_NPZ"
echo "Config:         ${CFG_PATH:-<not set>}"
echo "Benchmark root: $BENCHMARK_ROOT"
echo "Inference root: $INFERENCE_ROOT"
echo

if [ "$SKIP_RENDER" -eq 0 ]; then
    printf 'Render command: '
    printf '%q ' "${RENDER_CMD[@]}"
    printf '\n'
    if [ "$DRY_RUN" -eq 0 ]; then
        "${RENDER_CMD[@]}"
    fi
fi

if [ "$SKIP_INFERENCE" -eq 0 ]; then
    printf 'Inference command: '
    printf '%q ' "${INFER_CMD[@]}"
    printf '\n'
    if [ "$DRY_RUN" -eq 0 ]; then
        "${INFER_CMD[@]}"
    fi
fi

if [ "$SKIP_VIDEO_GRID" -eq 0 ]; then
    printf 'Video grid command: '
    printf '%q ' "${VIDEO_CMD[@]}"
    printf '\n'
    if [ "$DRY_RUN" -eq 0 ]; then
        "${VIDEO_CMD[@]}"
    fi
fi
