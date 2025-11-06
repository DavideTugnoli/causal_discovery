#!/bin/bash
#SBATCH --job-name=post-run-check
#SBATCH --partition=turing-wide
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/share/malelab/dtugnoli/rex_causal_discovery/hpc/demetra/logs/post_run_resume_%j.out
#SBATCH --error=/share/malelab/dtugnoli/rex_causal_discovery/hpc/demetra/logs/post_run_resume_%j.err

# Script da lanciare con una dipendenza afterany sul job principale per avvisarti e chiudere il ciclo di post-processing.

set -euo pipefail

DEFAULT_SESSION_ID="019a4abc-ac29-71f0-8381-732d06afe1ad"

# Permette override tramite variabili d'ambiente al momento della sottomissione (`--export`).
TARGET_SESSION_ID="${TARGET_SESSION_ID:-$DEFAULT_SESSION_ID}"
WATCH_JOB_ID="${WATCH_JOB_ID:-${PARENT_JOB_ID:-unknown}}"

if [[ -z "${TARGET_SESSION_ID}" ]]; then
    echo "Errore: TARGET_SESSION_ID non definito." >&2
    exit 1
fi

# Garantiamo che il binario codex sia raggiungibile senza assumere PATH statico.
NVM_BIN="/u/dtugnoli/.nvm/versions/node/v18.20.8/bin"
if ! command -v codex >/dev/null 2>&1; then
    export PATH="${NVM_BIN}:${PATH}"
fi
if ! command -v codex >/dev/null 2>&1; then
    echo "Errore: codex CLI non trovata (verifica PATH e installazione)." >&2
    exit 1
fi

PROMPT_DEFAULT=$(cat <<EOF
Il job collegato (${WATCH_JOB_ID}) si è concluso.
1) Controlla gli output in /share/malelab/dtugnoli/rex_causal_discovery/results.
2) Se mancano file attesi, esamina i log in /share/malelab/dtugnoli/rex_causal_discovery/hpc/demetra/logs (file .out/.err corrispondenti).
3) Identifica e risolvi l'errore, poi rilancia il job principale.
4) Dopo aver rilanciato il job, invoca nuovamente /share/malelab/dtugnoli/rex_causal_discovery/hpc/demetra/post_run_resume_codex.slurm con sbatch --dependency=afterany:<jobid> per continuare il ciclo.
EOF
)
PROMPT_TEMPLATE="${FOLLOWUP_PROMPT:-${PROMPT_DEFAULT}}"

echo "=== Avvio resume automatico per la sessione ${TARGET_SESSION_ID} ==="
printf '%s\n' "${PROMPT_TEMPLATE}" | codex exec resume \
    --ask-for-approval never \
    --sandbox workspace-write \
    "${TARGET_SESSION_ID}" \
    -
