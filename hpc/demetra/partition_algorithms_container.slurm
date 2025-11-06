#!/bin/bash
#SBATCH --job-name=partition-cd-container
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4
#SBATCH --output=/share/malelab/dtugnoli/rex_causal_discovery/hpc/demetra/logs/partition_algorithms_%A_%a.out
#SBATCH --error=/share/malelab/dtugnoli/rex_causal_discovery/hpc/demetra/logs/partition_algorithms_%A_%a.err

set -euo pipefail

REPO="/share/malelab/dtugnoli/rex_causal_discovery"
LOGDIR="$REPO/hpc/demetra/logs"
IMAGEDIR="$REPO/hpc/demetra/images"
SIF="$IMAGEDIR/causal_partition_algorithms.sif"
CONTAINER_URI="docker://ghcr.io/shahashka/causal_discovery_via_partitioning:main"
PYTHON_BIN="/opt/conda/envs/causal_discovery/bin/python"

IFS=' ' read -r -a ALGORITHMS <<< "${CAUSAL_ALGORITHMS:-GES PC RFCI NOTEARS RFCI-PAG}"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#ALGORITHMS[@]} )); then
    echo "Errore: SLURM_ARRAY_TASK_ID=${TASK_ID} fuori dal range [0, ${#ALGORITHMS[@]})." >&2
    exit 1
fi

CAUSAL_ALG="${ALGORITHMS[$TASK_ID]}"
CAUSAL_ALG_KEY=$(echo "$CAUSAL_ALG" | tr '[:upper:]' '[:lower:]')
CAUSAL_ALG_KEY=${CAUSAL_ALG_KEY//-/_}

DATA_PATH="${DATA_PATH:-dataset/health_subsets/health_lifestyle_500_numeric.csv}"
if [[ -z "${OUTPUT_TEMPLATE:-}" ]]; then
    OUTPUT_TEMPLATE="results/health_500_partition_{alg}_adj.csv"
fi
OUTPUT_PATH="${OUTPUT_TEMPLATE//\{alg\}/$CAUSAL_ALG_KEY}"
PC_ALPHA="${PC_ALPHA:-0.25}"
PARTITION_RES="${PARTITION_RES:-1.0}"
MERGE_FN="${MERGE_FN:-screen}"
USE_SKELETON="${USE_SKELETON:-1}"
FINITE_LIMIT="${FINITE_LIMIT:-1}"
MAX_WORKERS="${MAX_WORKERS:-${SLURM_CPUS_PER_TASK:-0}}"
USER_SITE_HOST="$REPO/.local/lib/python3.11/site-packages"
USER_SITE_CONTAINER="/workspace/.local/lib/python3.11/site-packages"
CACHE_DIR_HOST="$REPO/.cache"
CACHE_DIR_CONTAINER="/workspace/.cache"
HOST_SITE_SRC="/u/dtugnoli/.local"
HOST_SITE_CONTAINER="/host_local"
HOST_SITE_PYTHON="${HOST_SITE_CONTAINER}/lib/python3.11/site-packages"

mkdir -p "$LOGDIR" "$IMAGEDIR" "$REPO/results" "$USER_SITE_HOST" "$CACHE_DIR_HOST"

module purge
module load apptainer || module load singularity

echo "=== GPU visibility on host ==="
nvidia-smi || echo "nvidia-smi unavailable on host"

if [ ! -f "$SIF" ]; then
    echo "=== Scarico il container ($CONTAINER_URI) in $SIF ==="
    apptainer pull "$SIF" "$CONTAINER_URI"
fi

cd "$REPO"

CMD=("$PYTHON_BIN" /workspace/divide_et_impera/causal_discovery_via_partitioning/examples/partition_health_algorithms.py \
    --causal-alg "$CAUSAL_ALG" \
    --data-path "/workspace/$DATA_PATH" \
    --pc-alpha "$PC_ALPHA" \
    --partition-resolution "$PARTITION_RES" \
    --output-adj "/workspace/$OUTPUT_PATH")

if [[ "$MERGE_FN" != "screen" ]]; then
    CMD+=(--merge-fn "$MERGE_FN")
fi

if [[ "$USE_SKELETON" == "0" ]]; then
    CMD+=(--no-skeleton)
fi

if [[ "$FINITE_LIMIT" == "0" ]]; then
    CMD+=(--no-finite-limit)
fi

if [[ "$MAX_WORKERS" != "0" ]]; then
    CMD+=(--max-workers "$MAX_WORKERS")
fi

export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export APPTAINERENV_CUDA_HOME="${CUDA_HOME:-}"
export APPTAINERENV_TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
export APPTAINERENV_XDG_CACHE_HOME="${CACHE_DIR_CONTAINER}"
export APPTAINERENV_PYTHONPATH="/workspace/divide_et_impera/causal_discovery_via_partitioning:/workspace:${USER_SITE_CONTAINER}:${HOST_SITE_PYTHON}:${PYTHONPATH:-}"

echo "=== Comando nel container (algoritmo: $CAUSAL_ALG) ==="
printf '%q ' "${CMD[@]}"; echo

apptainer exec --nv \
    --home "$REPO:/workspace" \
    --bind "$HOST_SITE_SRC":"$HOST_SITE_CONTAINER":ro \
    --bind "$REPO":/workspace \
    "$SIF" \
    "${CMD[@]}"

echo "=== Job completato (algoritmo: $CAUSAL_ALG) ==="
