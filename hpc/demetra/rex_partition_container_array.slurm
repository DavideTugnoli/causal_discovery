#!/bin/bash
#SBATCH --job-name=rex-partition-array
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5
#SBATCH --output=/share/malelab/dtugnoli/rex_causal_discovery/hpc/demetra/logs/rex_partition_array_%A_%a.out
#SBATCH --error=/share/malelab/dtugnoli/rex_causal_discovery/hpc/demetra/logs/rex_partition_array_%A_%a.err

set -euo pipefail

REPO="/share/malelab/dtugnoli/rex_causal_discovery"
LOGDIR="$REPO/hpc/demetra/logs"
IMAGEDIR="$REPO/hpc/demetra/images"
SIF="$IMAGEDIR/causal_partition_rex.sif"
CONTAINER_URI="docker://ghcr.io/shahashka/causal_discovery_via_partitioning:main"
PYTHON_BIN="/opt/conda/envs/causal_discovery/bin/python"

declare -a DATASET_SIZES=(500 1000 2000 3000 4000 5000)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$TASK_ID" -lt 0 || "$TASK_ID" -ge "${#DATASET_SIZES[@]}" ]]; then
    echo "Array index ${TASK_ID} fuori range (0-${#DATASET_SIZES[@]})."
    exit 1
fi
SIZE="${DATASET_SIZES[$TASK_ID]}"
DATA_PATH="${DATA_PATH:-dataset/health_subsets/health_lifestyle_${SIZE}_numeric.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-results/health_${SIZE}_partition_rex_adj.csv}"
PC_ALPHA="${PC_ALPHA:-0.25}"
PARTITION_RES="${PARTITION_RES:-1.0}"
DEFAULT_MAX_WORKERS="${DEFAULT_MAX_WORKERS:-1}"
if [[ -z "${MAX_WORKERS:-}" ]]; then
    MAX_WORKERS="$DEFAULT_MAX_WORKERS"
fi
if [[ "${SLURM_CPUS_PER_TASK:-0}" -gt 0 && "$MAX_WORKERS" -gt "${SLURM_CPUS_PER_TASK:-0}" ]]; then
    MAX_WORKERS="${SLURM_CPUS_PER_TASK:-0}"
fi
OPTUNA_DB_FILE="results/optuna_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}_${SLURM_PROCID:-0}.db"
OPTUNA_DB_PATH="$REPO/$OPTUNA_DB_FILE"

if [[ ! -f "$REPO/$DATA_PATH" ]]; then
    echo "Dataset non trovato: $REPO/$DATA_PATH"
    exit 2
fi

module purge
module load apptainer || module load singularity

echo "=== GPU visibility on host ==="
nvidia-smi || echo "nvidia-smi unavailable on host"

if [ ! -f "$SIF" ]; then
    echo "=== Scarico il container ($CONTAINER_URI) in $SIF ==="
    apptainer pull "$SIF" "$CONTAINER_URI"
fi

cd "$REPO"

USER_SITE_HOST="$REPO/.local/lib/python3.11/site-packages"
USER_SITE_CONTAINER="/workspace/.local/lib/python3.11/site-packages"
CACHE_DIR_HOST="$REPO/.cache"
CACHE_DIR_CONTAINER="/workspace/.cache"
HOST_SITE_SRC="/u/dtugnoli/.local"
HOST_SITE_CONTAINER="/host_local"
HOST_SITE_PYTHON="${HOST_SITE_CONTAINER}/lib/python3.11/site-packages"

mkdir -p "$LOGDIR" "$IMAGEDIR" "$REPO/results" "$USER_SITE_HOST" "$CACHE_DIR_HOST"
rm -f "$OPTUNA_DB_PATH"

CMD=("$PYTHON_BIN" /workspace/divide_et_impera/causal_discovery_via_partitioning/examples/rex_health_partition.py \
    --data-path "/workspace/$DATA_PATH" \
    --output-adj "/workspace/$OUTPUT_PATH" \
    --pc-alpha "$PC_ALPHA" \
    --partition-resolution "$PARTITION_RES")

if [[ "$MAX_WORKERS" != "0" ]]; then
    CMD+=(--max-workers "$MAX_WORKERS")
fi

export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export APPTAINERENV_CUDA_HOME="${CUDA_HOME:-}"
export APPTAINERENV_TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
export APPTAINERENV_XDG_CACHE_HOME="${CACHE_DIR_CONTAINER}"
export APPTAINERENV_PYTHONPATH="/workspace/divide_et_impera/causal_discovery_via_partitioning:/workspace:${USER_SITE_CONTAINER}:${HOST_SITE_PYTHON}:${PYTHONPATH:-}"
export APPTAINERENV_OPTUNA_STORAGE="sqlite:////workspace/${OPTUNA_DB_FILE}"

echo "=== Comando nel container (dataset ${SIZE}) ==="
printf '%q ' "${CMD[@]}"; echo

apptainer exec --nv \
    --home "$REPO:/workspace" \
    --bind "$HOST_SITE_SRC":"$HOST_SITE_CONTAINER":ro \
    --bind "$REPO":/workspace \
    "$SIF" \
    "${CMD[@]}"

echo "=== Job completato per dataset ${SIZE} ==="
