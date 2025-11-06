# Demetra job scripts

Questa directory contiene gli script SLURM per lanciare gli esperimenti tramite Apptainer sul nodo Demetra.

## `partition_algorithms_container.slurm`

- Lancia tutti gli algoritmi di causal discovery implementati nella pipeline divide-et-impera (`GES`, `PC`, `RFCI`, `NOTEARS`, `RFCI-PAG`) usando un job array (`--array=0-4`).
- Per eseguire: `sbatch partition_algorithms_container.slurm` (o `sbatch --array=0-4 partition_algorithms_container.slurm` se vuoi specificarlo esplicitamente).
- Variabili d'ambiente utili:
  - `CAUSAL_ALGORITHMS`: lista separata da spazi per personalizzare gli algoritmi (es. `"GES PC NOTEARS"`). Ricorda di aggiornare anche `--array`.
  - `OUTPUT_TEMPLATE`: pattern per gli output (`{alg}` verrà sostituito con il nome normalizzato), default `results/health_500_partition_{alg}_adj.csv`.
  - `DATA_PATH`, `PC_ALPHA`, `PARTITION_RES`, `MAX_WORKERS`, `MERGE_FN`, `USE_SKELETON`, `FINITE_LIMIT` per controllare dataset, parametri di partizionamento, parallelismo e fusion.
- All'interno del container viene eseguito `/workspace/divide_et_impera/causal_discovery_via_partitioning/examples/partition_health_algorithms.py`, che salva l'adiacenza finale in CSV (uno per algoritmo).

Gli script esistenti per ReX (`rex_partition_container*.slurm`) rimangono invariati e continuano a funzionare come prima.
