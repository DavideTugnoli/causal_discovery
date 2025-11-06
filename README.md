# REX + Divide-and-Conquer Causal Discovery

This repository now contains an *integration workflow* that combines the
[`causalgraph`](./causalgraph) implementation of **REX** with the divide and conquer
framework from [`causal_discovery_via_partitioning`](./causal_discovery_via_partitioning).
The goal is to obtain high-recall causal skeletons (and optional orientations) on
large tabular datasets such as `dataset/simglucose_static_scm_with_params_6000.csv`.

The central artefact is the script [`rex_dc.py`](./rex_dc.py). It glues together the
two codebases as follows:

1. **REX superstructure**  
   REX trains both the neural and gradient-boosting regressors, runs the
   bootstrapped SHAP-based parent selection, and keeps edges whose selection
   frequency is above a low threshold `τ`. The union of the two regressors’
   selections produces a dense skeleton that maximises edge recall. The neural
   regressor can be trained on GPU via `--nn-accelerator`, while SHAP-based
   attribution deliberately stays on CPU to avoid device conflicts.

2. **Causal partition**  
   The skeleton is handed to the divide-and-conquer library to create an
   overlapping partition (edge-cover by default, expansive-causal and modularity
   are also available). Each subset contains a small number of variables, which
   keeps the local causal discovery tasks tractable.

3. **Local learning + merge**  
   For every partition, a causal discovery learner (default: RFCI-PAG via
   `pcalg`) is executed. The local outputs are then merged with the “screening”
   algorithm that returns a global CPDAG whose edges are guaranteed to lie within
   the REX superstructure. This step is embarrassingly parallel and operates on
   modestly sized subgraphs, which makes it suitable for HPC deployments.

The script emits the superstructure, the partitions and the final global graph,
ready to be copied back to the HPC cluster for large-scale runs.

---

## Installation

> These commands assume you use `uv` for Python environments, as mentioned in the
> original instructions. Any other environment manager works as long as the two
> libraries are installed in editable mode.

```bash
uv venv
uv pip install -e ./causalgraph
uv pip install -e ./causal_discovery_via_partitioning
```

For RFCI/PAG learning you also need `R`, `rpy2`, and the `pcalg` package (see
[`requirements.R`](./causal_discovery_via_partitioning/requirements.R) or reuse
the provided Dockerfile on the HPC system).

---

## Running the pipeline

```
uv run python rex_dc.py \
  --csv dataset/simglucose_static_scm_with_params_6000.csv \
  --out results \
  --bootstraps 30 \
  --tau 0.15 \
  --partition edge_cover \
  --local RFCI-PAG \
  --subset 20 \
  --nn-accelerator auto
```

Use `--nn-accelerator auto` (or `gpu`) to offload the NN regressor to a GPU when
one is available; omit the flag or set it to `cpu` to keep the training fully on
CPU. Add `--auto-params` if you prefer the script to pick `τ`, bootstraps,
sampling split, parallelism, and partition sizes automatically from the dataset
shape and your hardware budget.

Key command-line flags:

- `--csv` – path to the input dataset (columns = variables).
- `--out` – output directory (created if it does not exist).
- `--bootstraps` – number of REX bootstrap iterations (higher boosts stability).
- `--tau` – selection threshold; smaller values keep more candidate edges.
- `--partition` – overlap strategy: `edge_cover`, `expansive_causal`, or
  `modularity`.
- `--local` – causal learner per subset: `RFCI-PAG`, `PC`, `GES`, or `NOTEARS`.
- `--subset` – target number of variables per partition (tunes parallel workload).
- `--nn-accelerator` – accelerator for the NN regressor (`cpu`, `auto`,
  `gpu`/`cuda`, `mps`); SHAP computations always run on CPU.
- `--auto-params` – enable the resource-aware auto-tuner (chooses `τ`,
  bootstraps, sampling split, parallelism, accelerator, and partition size).
- `--max-ram-frac`, `--pilot-bootstraps` – refine the auto-tuner (RAM budget
  per SHAP iteration and number of pilot bootstraps respectively).
- `--sampling-split` and `--parallel-jobs` – advanced REX bootstrap controls.

### Auto-tuning at a glance

When `--auto-params` is enabled the script:

- runs a short “pilot” (default 3 bootstraps) to estimate edge frequencies;
- selects `τ` so the average degree of the REX superstructure lands in a
  recall-friendly band (≈10–25 % of the variables, capped at 3–12 neighbours);
- scales the number of bootstraps with `n·p` (40/30/20/12);
- lowers `sampling_split` to stay within `--max-ram-frac` of available RAM;
- enables parallel bootstraps only when ≥24 GB are free, and chooses the NN
  accelerator (`cpu`/`auto`/`gpu`/`mps`), retrying on CPU automatically if a
  GPU/MPS run exhausts memory;
- sizes the partition subsets to roughly `10 + 2√p` (clamped to 15–25) for
  stable local learning.

The script stores:

- `rex_superstructure.csv` – binary adjacency matrix of the REX skeleton.
- `global_graph.edgelist` – edge list of the merged CPDAG/DAG.
- `partition.json` – JSON description of the overlapping partitions.
- `columns.txt` – ordered variable names (aligned with the adjacency matrices).

---

## How it works internally

Each stage of the pipeline is exposed as a standalone function:

- `rex_superstructure(df, ...)` – returns the union skeleton and column order.
- `build_partition(superstructure, df_with_target, ...)` – builds a causal partition.
- `run_partitioned_cd(df, superstructure, partition, ...)` – executes the local
  learner and merges the results.
- `rex_divide_and_conquer(df, ...)` – orchestrates the full workflow and returns a
  dataclass with all intermediate artefacts.

This design allows you to experiment interactively (e.g. in a notebook on the
HPC cluster) by composing the pieces you need.

---

## Notes for HPC deployment

- By default the workload is CPU-bound, but `--nn-accelerator auto` (or `gpu`)
  moves the NN fitting to the GPU while SHAP stays on CPU; combine this with
  `--parallel-jobs` or job-level parallelism for larger datasets.
- The auto-tuner (`--auto-params`) keeps memory usage in check via
  `--max-ram-frac`; override the fraction if your HPC nodes have different RAM
  budgets.
- The divide-and-conquer stage is embarrassingly parallel across partitions; on
  the HPC you can split the partition dictionary and launch multiple processes.
- For extremely large datasets you can run `rex_superstructure` separately,
  serialise the adjacency, and reuse it in multiple partitioning/merging runs.

---

## Next steps

- Tune `τ`, the number of bootstraps, and the partition size to match each
  dataset’s scale and the available GPU/CPU budget on the HPC cluster.
- If orientations are not needed, you can stop after the REX superstructure and
  still benefit from the high-recall skeleton.
- To mix in additional skeletons (e.g. a liberal PC run), take the element-wise
  union before building the partition.
