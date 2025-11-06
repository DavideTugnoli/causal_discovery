#!/bin/bash
#SBATCH --job-name=auto-check
#SBATCH --partition=turing-wide
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/share/malelab/dtugnoli/rex_causal_discovery/hpc/demetra/logs/auto_resume_%j.out
#SBATCH --error=/share/malelab/dtugnoli/rex_causal_discovery/hpc/demetra/logs/auto_resume_%j.err

# Script da lanciare con dipendenza afterany sul job principale per analisi e rilancio automatico.

set -euo pipefail

DEFAULT_SESSION_ID="11f90cfc-9571-4f0b-99aa-2783c1e444e1"

# Permette override tramite variabili d'ambiente al momento della sottomissione (`--export`).
TARGET_SESSION_ID="${TARGET_SESSION_ID:-$DEFAULT_SESSION_ID}"
WATCH_JOB_ID="${WATCH_JOB_ID:-${PARENT_JOB_ID:-unknown}}"

if [[ -z "${TARGET_SESSION_ID}" ]]; then
    echo "Errore: TARGET_SESSION_ID non definito." >&2
    exit 1
fi

# Garantiamo che il binario claude sia raggiungibile senza assumere PATH statico.
# Adatta il path se hai installato claude altrove.
if ! command -v claude >/dev/null 2>&1; then
    # Tenta path comuni (adatta se necessario)
    for bin_path in \
        "$HOME/.local/bin" \
        "$HOME/bin" \
        "/usr/local/bin" \
        "$HOME/.nvm/versions/node/v18.20.8/bin"; do
        if [[ -d "$bin_path" ]] && [[ -x "$bin_path/claude" ]]; then
            export PATH="$bin_path:$PATH"
            break
        fi
    done
fi

if ! command -v claude >/dev/null 2>&1; then
    echo "Errore: CLI claude non trovata (verifica PATH e installazione)." >&2
    exit 1
fi

PROMPT_DEFAULT=$(cat <<EOF
Il job collegato (${WATCH_JOB_ID}) si è concluso.

Analizza automaticamente la situazione:
1) Controlla gli output in /share/malelab/dtugnoli/rex_causal_discovery/results/
2) Se mancano file attesi, esamina i log in /share/malelab/dtugnoli/rex_causal_discovery/hpc/demetra/logs/ (file .out/.err corrispondenti al job ${WATCH_JOB_ID})
3) Identifica e risolvi l'errore trovato (modifica codice se necessario)
4) Rilancia il job principale con sbatch e gli stessi parametri
5) Dopo aver rilanciato, invoca nuovamente questo script con:
   sbatch --dependency=afterany:<nuovo_jobid> --export=WATCH_JOB_ID=<nuovo_jobid> /share/malelab/dtugnoli/rex_causal_discovery/hpc/demetra/post_run_resume_claude.slurm

Procedi autonomamente senza chiedere conferme.
EOF
)
PROMPT_TEMPLATE="${FOLLOWUP_PROMPT:-${PROMPT_DEFAULT}}"

echo "=== Avvio resume automatico per la sessione ${TARGET_SESSION_ID} ==="
printf '%s\n' "${PROMPT_TEMPLATE}" | claude \
    --resume "${TARGET_SESSION_ID}" \
    -p \
    --dangerously-skip-permissions \
    --output-format text
