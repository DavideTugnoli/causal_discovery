#!/bin/bash
#SBATCH --job-name=rex_sif_pull
#SBATCH --partition=turing-long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/share/malelab/dtugnoli/rex_causal_discovery/hpc/demetra/logs/rex_sif_pull_%j.out
#SBATCH --error=/share/malelab/dtugnoli/rex_causal_discovery/hpc/demetra/logs/rex_sif_pull_%j.err

set -euo pipefail

REPO_ROOT=/share/malelab/dtugnoli/rex_causal_discovery
IMG_DIR="${REPO_ROOT}/hpc/demetra/images"
CACHE_DIR="${REPO_ROOT}/hpc/demetra/apptainer_cache"
TMP_DIR="${SLURM_TMPDIR:-${CACHE_DIR}/tmp/${SLURM_JOB_ID}}"

mkdir -p "${IMG_DIR}" "${CACHE_DIR}" "${TMP_DIR}"

module purge
module load apptainer

export APPTAINER_CACHEDIR="${CACHE_DIR}"
export SINGULARITY_CACHEDIR="${CACHE_DIR}"
export APPTAINER_TMPDIR="${TMP_DIR}"
export SINGULARITY_TMPDIR="${TMP_DIR}"
export APPTAINER_MKSQUASHFS_PROCS="${SLURM_CPUS_PER_TASK:-8}"
export SINGULARITY_MKSQUASHFS_PROCS="${APPTAINER_MKSQUASHFS_PROCS}"

apptainer pull --force \
    "${IMG_DIR}/causal_partition_rex.sif" \
    docker://ghcr.io/shahashka/causal_discovery_via_partitioning:main
