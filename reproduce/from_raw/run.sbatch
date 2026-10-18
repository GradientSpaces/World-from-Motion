#!/bin/bash
#SBATCH --job-name=wfm_independent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=03:59:00
#SBATCH --array=0-13%2
set -euo pipefail

: "${WFM_REPO_ROOT:?Set the release checkout}"
: "${WFM_RAW_INPUTS:?Set the verified download directory}"
: "${WFM_RAW_OUTPUT:?Set a fresh experiment root}"
SCENES=(apple block paper-windmill space-out spin teddy wheel)
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit an array; wheel pilots are indices 6,13}"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID > 13 )); then exit 2; fi
METHOD=ms
if (( TASK_ID >= 7 )); then METHOD=wt; fi
SCENE="${SCENES[$((TASK_ID % 7))]}"
CMD=("${WFM_PYTHON:-python}" "$WFM_REPO_ROOT/reproduce/from_raw/run.py"
     --inputs "$WFM_RAW_INPUTS" --output "$WFM_RAW_OUTPUT/${METHOD}_${SCENE}"
     --method "$METHOD" --scene "$SCENE" --cpu-threads "${SLURM_CPUS_PER_TASK:-16}")
printf -v CMD_TEXT '%q ' "${CMD[@]}"
if [[ -n "${WFM_CONDA_ACTIVATE:-}" ]]; then
    printf -v CMD_TEXT 'source %q %q && export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}" && exec %s' \
        "$WFM_CONDA_ACTIVATE" "${WFM_CONDA_ENV:-mosca}" "$CMD_TEXT"
fi
SRUN=(srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}")
if [[ -n "${WFM_CONTAINER_IMAGE:-}" ]]; then
    : "${WFM_CONTAINER_MOUNTS:?Specify mounts for your cluster}"
    SRUN+=(--container-image="$WFM_CONTAINER_IMAGE" --container-mounts="$WFM_CONTAINER_MOUNTS"
           --container-workdir="$WFM_REPO_ROOT" --no-container-entrypoint)
fi
"${SRUN[@]}" bash -c "$CMD_TEXT"
