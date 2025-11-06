"""
High-recall causal discovery pipeline that combines REX and the divide & conquer
causal partition framework.

Steps
-----
1. Run REX with bootstrapping and a low selection threshold to obtain a dense
   superstructure (skeleton) that prioritises recall.
2. Build an overlapping causal partition on that superstructure (edge-cover by
   default, or expansive-causal/modularity).
3. Run a local causal discovery algorithm (RFCI-PAG by default) on each
   subproblem.
4. Merge the local graphs with the screening procedure to obtain a global CPDAG.

The script is intended to be executed after installing the local checkouts of
`causalgraph` and `causal_discovery_via_partitioning`, e.g. with:

    uv pip install -e ./causalgraph
    uv pip install -e ./causal_discovery_via_partitioning

Outputs
-------
* `rex_superstructure.csv` : binary adjacency matrix of the REX union skeleton.
* `global_graph.edgelist`  : edge list of the merged global graph (CPDAG/DAG).
* `partition.json`         : JSON description of the node partitions.
* `columns.txt`            : ordered list of variable names.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_regression

from causalexplain.estimators.rex.rex import Rex

from cd_v_partition.overlapping_partition import (
    expansive_causal_partition,
    modularity_partition,
    partition_problem,
    rand_edge_cover_partition,
)
from cd_v_partition.causal_discovery import (
    dagma_local_learn,
    ges_local_learn,
    pc_local_learn,
    pc as pc_wrapper,
    rfci_pag_local_learn,
)
from cd_v_partition.fusion import (
    remove_edges_not_in_ss,
    screen_projections,
    screen_projections_pag2cpdag,
)


# --------------------------------------------------------------------------- #
# 1) REX-based superstructure                                                #
# --------------------------------------------------------------------------- #


def _resolve_nn_accelerator(requested: str | None) -> str:
    """
    Normalise the accelerator string for the NN regressor and validate availability.
    """
    if requested is None:
        return "cpu"

    req = requested.lower()
    valid = {"cpu", "auto", "gpu", "cuda", "mps"}
    if req not in valid:
        raise ValueError(
            f"Invalid nn_accelerator '{requested}'. "
            "Valid options are: cpu, auto, gpu/cuda, mps."
        )

    if req in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU accelerator requested but torch.cuda.is_available() is False."
            )
        return "gpu"

    if req == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS accelerator requested but not available.")
        return "mps"

    if req == "auto":
        if torch.cuda.is_available():
            return "gpu"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    return "cpu"


def _fit_rex_model(
    name: str,
    model_type: str,
    data: pd.DataFrame,
    *,
    nn_device: str | None = None,
    verbose: bool = False,
) -> Rex:
    """Fit a REX estimator for a given regressor type."""
    rex = Rex(
        name=name,
        model_type=model_type,
        tune_model=False,
        verbose=verbose,
        prog_bar=False,
    )
    pipeline = None
    if model_type == "nn" and nn_device is not None:
        pipeline = [
            ('models', rex.model_type, {'device': nn_device}),
            ('models.tune_fit', {'hpo_n_trials': rex.hpo_n_trials}),
            ('models.score', {})
        ]
    rex.fit(data, pipeline=pipeline)
    return rex


def _available_ram_bytes() -> int:
    """Best-effort detection of available RAM (fallback to 16GB)."""
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb * 1024
        except Exception:
            pass
    return 16 * 1024**3


def _pilot_union_freqs(
    data: pd.DataFrame,
    pilot_bootstraps: int,
    sampling_split: float,
    nn_accelerator: str,
    random_state: int = 1234,
) -> np.ndarray:
    """Run a tiny pilot for NN and GBT; return symmetric frequency matrix."""
    p = data.shape[1]
    freqs = np.zeros((p, p), dtype=float)
    for model in ("nn", "gbt"):
        device = nn_accelerator if model == "nn" else None
        try:
            rex = _fit_rex_model(
                f"pilot_{model}",
                model,
                data,
                nn_device=device,
                verbose=False,
            )
        except RuntimeError as err:
            msg = str(err)
            if "CUDA out of memory" in msg or "MPS" in msg:
                rex = _fit_rex_model(
                    f"pilot_{model}",
                    model,
                    data,
                    nn_device=None,
                    verbose=False,
                )
            else:
                raise

        adjacency = rex._build_bootstrapped_adjacency_matrix(  # type: ignore[attr-defined]
            data,
            num_iterations=pilot_bootstraps,
            sampling_split=sampling_split,
            prior=None,
            parallel_jobs=0,
            random_state=random_state,
        )
        adjacency = adjacency / max(1, pilot_bootstraps)
        freqs = np.maximum(freqs, adjacency)

    freqs = np.maximum(freqs, freqs.T)
    np.fill_diagonal(freqs, 0.0)
    return freqs


def _avg_degree_from_freq(freqs: np.ndarray, tau: float) -> float:
    binary = (freqs >= tau).astype(int)
    skeleton = np.maximum(binary, binary.T)
    np.fill_diagonal(skeleton, 0)
    return float(skeleton.sum(axis=1).mean())


@dataclass
class AutoParams:
    bootstraps: int
    tau: float
    sampling_split: float
    parallel_jobs: int
    nn_accelerator: str
    target_subset_size: int
    avg_degree: float


def auto_select_params(
    data: pd.DataFrame,
    *,
    prefer_recall: bool = True,
    max_ram_frac: float = 0.35,
    pilot_bootstraps: int = 3,
    tau_grid: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    nn_accelerator_hint: str = "auto",
) -> AutoParams:
    """
    Choose REX+D&C knobs from data size and resources.

    - memory-bounds sampling_split (SHAP CPU memory),
    - scale bootstraps with n*p,
    - pick tau via a cheap pilot to target a safe/high-recall skeleton density,
    - size partition subsets for local learners,
    - keep parallelism conservative unless RAM is abundant.
    """
    n, p = data.shape
    available_ram = _available_ram_bytes()
    mem_budget = int(available_ram * max_ram_frac)

    bytes_per_value = 8.0
    overhead = 6.0
    n_sub_max = int(mem_budget / max(1.0, overhead * p * bytes_per_value))
    sampling_split = max(0.15, min(0.66, n_sub_max / max(1, n)))

    scale = n * p
    if scale <= 300_000:
        bootstraps = 40
    elif scale <= 1_000_000:
        bootstraps = 30
    elif scale <= 2_000_000:
        bootstraps = 20
    else:
        bootstraps = 12

    try:
        nn_accelerator = _resolve_nn_accelerator(nn_accelerator_hint)
    except Exception:
        nn_accelerator = "cpu"

    freqs = _pilot_union_freqs(
        data,
        pilot_bootstraps=pilot_bootstraps,
        sampling_split=sampling_split,
        nn_accelerator=nn_accelerator,
    )

    k_low = max(3, int(0.10 * p))
    k_high = min(12, int(0.25 * p))

    chosen_tau: Optional[float] = None
    chosen_deg: Optional[float] = None

    for t in tau_grid:
        degree = _avg_degree_from_freq(freqs, t)
        if k_low <= degree <= k_high:
            chosen_tau = t
            chosen_deg = degree
            break

    if chosen_tau is None:
        candidates = []
        for t in tau_grid:
            degree = _avg_degree_from_freq(freqs, t)
            # Prefer lower tau (higher recall) on ties if prefer_recall is True.
            recall_bias = -t if prefer_recall else t
            candidates.append((abs(degree - k_high), recall_bias, t, degree))
        candidates.sort()
        _, _, chosen_tau, chosen_deg = candidates[0]

    target_subset_size = int(max(15, min(25, round(10 + 2 * np.sqrt(p)))))
    parallel_jobs = -1 if mem_budget >= 24 * 1024**3 else 0

    return AutoParams(
        bootstraps=int(bootstraps),
        tau=float(chosen_tau),
        sampling_split=float(sampling_split),
        parallel_jobs=int(parallel_jobs),
        nn_accelerator=str(nn_accelerator),
        target_subset_size=target_subset_size,
        avg_degree=float(chosen_deg),
    )


def rex_superstructure(
    data: pd.DataFrame,
    *,
    bootstraps: int = 30,
    tau: float = 0.15,
    models: Sequence[str] = ("nn", "gbt"),
    sampling_split: float = 0.66,
    parallel_jobs: int = 0,
    random_state: int = 1234,
    nn_accelerator: str = "cpu",
    verbose: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a dense, high-recall undirected superstructure using the union of
    bootstrapped REX adjacencies across the requested regressors. The neural
    regressor can leverage GPU/MPS via `nn_accelerator`, whereas SHAP
    computations remain on CPU.
    """
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1].")

    columns = list(data.columns)
    n_features = len(columns)
    superstructure = np.zeros((n_features, n_features), dtype=int)
    nn_device = None
    if any(model == "nn" for model in models):
        nn_device = _resolve_nn_accelerator(nn_accelerator)

    for model_type in models:
        device_arg = nn_device if model_type == "nn" else None
        rex = _fit_rex_model(
            f"rex_{model_type}",
            model_type,
            data,
            nn_device=device_arg,
            verbose=verbose,
        )
        adjacency = rex._build_bootstrapped_adjacency_matrix(  # type: ignore[attr-defined]
            data,
            num_iterations=bootstraps,
            sampling_split=sampling_split,
            prior=None,
            parallel_jobs=parallel_jobs,
            random_state=random_state,
        )
        binary_adj = (adjacency >= tau).astype(int)
        skeleton = np.maximum(binary_adj, binary_adj.T)
        np.fill_diagonal(skeleton, 0)
        superstructure = np.maximum(superstructure, skeleton)

        if device_arg in {"gpu"} or (device_arg == "auto" and torch.cuda.is_available()):
            torch.cuda.empty_cache()

    return superstructure, columns


# --------------------------------------------------------------------------- #
# PC-based superstructure + helper utilities                                  #
# --------------------------------------------------------------------------- #


def _pairwise_mi_matrix(
    X: np.ndarray,
    *,
    n_neighbors: int = 3,
    random_state: int = 0,
) -> np.ndarray:
    """
    Estimate a symmetric pairwise mutual-information matrix for continuous variables.
    Uses k-NN MI estimates for each pair (i, j).
    """
    p = X.shape[1]
    mi_matrix = np.zeros((p, p), dtype=float)
    rng = np.random.RandomState(random_state)
    for i in range(p):
        for j in range(i + 1, p):
            # mutual_info_regression expects 2D features; evaluate both (i -> j) and mirror
            mi = mutual_info_regression(
                X[:, [i]],
                X[:, j],
                random_state=rng,
                n_neighbors=max(1, n_neighbors),
            )[0]
            mi_matrix[i, j] = mi_matrix[j, i] = float(mi)
    return mi_matrix


def pc_superstructure(
    data: pd.DataFrame,
    *,
    alpha: float = 0.2,
    ensure_min_degree: int = 1,
    mi_neighbors: int = 3,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a superstructure with the PC algorithm and optionally densify isolated nodes
    via mutual-information-based edges to preserve recall.
    """
    data_with_target = _ensure_target_column(data)
    p = data.shape[1]
    full_skel = np.ones((p, p), dtype=int)
    np.fill_diagonal(full_skel, 0)
    pdag, _ = pc_wrapper(
        data_with_target,
        full_skel,
        outdir=None,
        alpha=alpha,
    )
    pdag_np = np.array(pdag, dtype=float)
    skeleton = (pdag_np > 0).astype(int)
    skeleton = np.maximum(skeleton, skeleton.T)
    np.fill_diagonal(skeleton, 0)

    if ensure_min_degree > 0:
        degrees = skeleton.sum(axis=1)
        if degrees.min() < ensure_min_degree:
            X = data.to_numpy(dtype=float)
            mi_matrix = _pairwise_mi_matrix(
                X, n_neighbors=mi_neighbors, random_state=0
            )
            for node in range(p):
                while degrees[node] < ensure_min_degree:
                    candidates = np.argsort(mi_matrix[node])[::-1]
                    added = False
                    for cand in candidates:
                        if cand == node or skeleton[node, cand] == 1:
                            continue
                        skeleton[node, cand] = skeleton[cand, node] = 1
                        degrees[node] += 1
                        degrees[cand] += 1
                        added = True
                        break
                    if not added:
                        break

    return skeleton, list(data.columns)


def rex_local_learn(
    subproblem: Tuple[np.ndarray, pd.DataFrame],
    *,
    bootstraps: int = 12,
    tau: float = 0.10,
    models: Sequence[str] = ("gbt",),
    sampling_split: float = 0.66,
    parallel_jobs: int = 0,
    nn_accelerator: str = "cpu",
    verbose: bool = False,
) -> np.ndarray:
    """
    Run REX locally on a subproblem (subgraph) to obtain a skeleton adjacency matrix.
    """
    local_skel, data_with_target = subproblem
    local_df = data_with_target.drop(columns=["target"])
    nn_device = _resolve_nn_accelerator(nn_accelerator)
    ss_local, _ = rex_superstructure(
        local_df,
        bootstraps=bootstraps,
        tau=tau,
        models=models,
        sampling_split=sampling_split,
        parallel_jobs=parallel_jobs,
        nn_accelerator=nn_device,
        verbose=verbose,
    )
    ss_local = (ss_local > 0).astype(int)
    ss_local = np.minimum(ss_local, (local_skel > 0).astype(int))
    np.fill_diagonal(ss_local, 0)
    return ss_local


# --------------------------------------------------------------------------- #
# 2) Partitioning                                                            #
# --------------------------------------------------------------------------- #


PartitionDict = Dict[int, Iterable[int]]


def build_partition(
    superstructure: np.ndarray,
    data_with_target: pd.DataFrame,
    *,
    algo: str = "edge_cover",
    target_subset_size: int = 20,
    resolution: int = 5,
) -> PartitionDict:
    """
    Build an overlapping causal partition over the superstructure.

    Parameters
    ----------
    superstructure : np.ndarray
        Binary adjacency matrix (N x N).
    data_with_target : pd.DataFrame
        Dataset containing the variables plus a trailing `target` column.
    algo : str
        Partitioning algorithm: 'edge_cover', 'expansive_causal', or 'modularity'.
    target_subset_size : int
        Desired upper bound on the number of variables per subset.
    resolution : int
        Resolution parameter for the greedy modularity community detection.
    """
    n_vars = superstructure.shape[0]
    if target_subset_size <= 0:
        raise ValueError("target_subset_size must be positive.")

    best_n = max(1, math.ceil(n_vars / target_subset_size))
    cutoff = best_n

    if algo == "edge_cover":
        partition = rand_edge_cover_partition(
            superstructure,
            data=data_with_target,
            resolution=resolution,
            cutoff=cutoff,
            best_n=best_n,
        )
    elif algo == "expansive_causal":
        partition = expansive_causal_partition(
            superstructure,
            data=data_with_target,
            resolution=resolution,
            cutoff=cutoff,
            best_n=best_n,
        )
    elif algo == "modularity":
        partition = modularity_partition(
            superstructure,
            data=data_with_target,
            resolution=resolution,
            cutoff=cutoff,
            best_n=best_n,
        )
    else:
        raise ValueError(
            f"Unknown partition algorithm '{algo}'. "
            "Choose from {'edge_cover', 'expansive_causal', 'modularity'}."
        )

    return partition


# --------------------------------------------------------------------------- #
# 3) Local learning + merging                                                #
# --------------------------------------------------------------------------- #


def _ensure_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has a trailing 'target' column required by the
    divide-and-conquer codebase.
    """
    if "target" in df.columns:
        reordered = df.copy()
        cols = [c for c in reordered.columns if c != "target"] + ["target"]
        return reordered.loc[:, cols]

    result = df.copy()
    result["target"] = 0.0
    return result


def run_partitioned_cd(
    data: pd.DataFrame,
    superstructure: np.ndarray,
    partition: PartitionDict,
    *,
    local: str = "RFCI-PAG",
    finite_sample_screen: bool = True,
    local_rex_kwargs: Optional[Dict[str, object]] = None,
) -> nx.DiGraph:
    """
    Execute local causal discovery on each partition and merge the results.

    Returns a CPDAG if RFCI-PAG is selected; otherwise a directed graph.
    """
    data_with_target = _ensure_target_column(data)
    subproblems = partition_problem(partition, superstructure, data_with_target)

    local_results: List[np.ndarray] = []
    local_rex_kwargs = local_rex_kwargs or {}
    for subproblem in subproblems:
        if local == "RFCI-PAG":
            local_adj = rfci_pag_local_learn(subproblem, use_skel=True)
        elif local == "PC":
            local_adj = pc_local_learn(subproblem, use_skel=True)
        elif local == "GES":
            local_adj = ges_local_learn(subproblem, use_skel=True)
        elif local == "NOTEARS":
            local_adj = dagma_local_learn(subproblem, use_skel=True)
        elif local == "REX-SKEL":
            local_adj = rex_local_learn(subproblem, **local_rex_kwargs)
        else:
            raise ValueError(
                f"Unknown local learner '{local}'. "
                "Choose from {'RFCI-PAG', 'PC', 'GES', 'NOTEARS', 'REX-SKEL'}."
            )
        local_results.append(local_adj)

    observational_data = data_with_target.drop(columns=["target"]).to_numpy()
    ss_graph = nx.from_numpy_array(superstructure, create_using=nx.DiGraph)

    if local == "RFCI-PAG":
        global_graph = screen_projections_pag2cpdag(
            superstructure,
            partition,
            local_results,
            ss_subset=True,
            finite_lim=finite_sample_screen,
            data=observational_data,
        )
    else:
        global_graph = screen_projections(
            superstructure,
            partition,
            local_results,
            ss_subset=True,
            finite_lim=finite_sample_screen,
            data=observational_data,
        )

    return remove_edges_not_in_ss(global_graph, ss_graph)


# --------------------------------------------------------------------------- #
# 4) High-level convenience function                                         #
# --------------------------------------------------------------------------- #


@dataclass
class RexDivideAndConquerResult:
    superstructure: np.ndarray
    columns: List[str]
    partition: PartitionDict
    graph: nx.DiGraph


def rex_divide_and_conquer(
    data: pd.DataFrame,
    *,
    bootstraps: int = 30,
    tau: float = 0.15,
    partition_algo: str = "edge_cover",
    local: str = "RFCI-PAG",
    target_subset_size: int = 20,
    sampling_split: float = 0.66,
    parallel_jobs: int = 0,
    nn_accelerator: str = "cpu",
    verbose: bool = False,
    local_rex_kwargs: Optional[Dict[str, object]] = None,
) -> RexDivideAndConquerResult:
    """
    Complete end-to-end pipeline returning the REX superstructure, the
    partition, and the merged global graph. Pass `nn_accelerator` to control
    whether the neural regressor trains on CPU, GPU, or MPS (SHAP stays on CPU).
    """
    superstructure, columns = rex_superstructure(
        data,
        bootstraps=bootstraps,
        tau=tau,
        models=("nn", "gbt"),
        sampling_split=sampling_split,
        parallel_jobs=parallel_jobs,
        nn_accelerator=nn_accelerator,
        verbose=verbose,
    )

    partition_input = _ensure_target_column(data)
    partition = build_partition(
        superstructure,
        partition_input,
        algo=partition_algo,
        target_subset_size=target_subset_size,
    )

    global_graph = run_partitioned_cd(
        data,
        superstructure,
        partition,
        local=local,
        finite_sample_screen=True,
        local_rex_kwargs=local_rex_kwargs,
    )

    return RexDivideAndConquerResult(
        superstructure=superstructure,
        columns=columns,
        partition=partition,
        graph=global_graph,
    )


def pc_then_rex_divide_and_conquer(
    data: pd.DataFrame,
    *,
    pc_alpha: float = 0.2,
    ensure_min_degree: int = 1,
    mi_neighbors: int = 3,
    partition_algo: str = "edge_cover",
    target_subset_size: int = 20,
    local: str = "REX-SKEL",
    local_rex_kwargs: Optional[Dict[str, object]] = None,
    finite_sample_screen: bool = True,
) -> RexDivideAndConquerResult:
    """
    Pipeline variant: liberal PC superstructure + divide & conquer with local REX.
    """
    superstructure, columns = pc_superstructure(
        data,
        alpha=pc_alpha,
        ensure_min_degree=ensure_min_degree,
        mi_neighbors=mi_neighbors,
    )
    partition_input = _ensure_target_column(data)
    partition = build_partition(
        superstructure,
        partition_input,
        algo=partition_algo,
        target_subset_size=target_subset_size,
    )
    global_graph = run_partitioned_cd(
        data,
        superstructure,
        partition,
        local=local,
        finite_sample_screen=finite_sample_screen,
        local_rex_kwargs=local_rex_kwargs,
    )
    return RexDivideAndConquerResult(
        superstructure=superstructure,
        columns=columns,
        partition=partition,
        graph=global_graph,
    )


# --------------------------------------------------------------------------- #
# 5) CLI                                                                     #
# --------------------------------------------------------------------------- #


def _write_outputs(
    result: RexDivideAndConquerResult,
    output_dir: str,
    *,
    superstructure_prefix: str = "rex",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(
        os.path.join(output_dir, f"{superstructure_prefix}_superstructure.csv"),
        result.superstructure,
        fmt="%d",
        delimiter=",",
    )
    nx.write_edgelist(
        result.graph,
        os.path.join(output_dir, "global_graph.edgelist"),
        data=False,
    )
    with open(os.path.join(output_dir, "columns.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(result.columns))
    with open(os.path.join(output_dir, "partition.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {str(k): list(map(int, v)) for k, v in result.partition.items()},
            fh,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to the CSV dataset.")
    parser.add_argument(
        "--out",
        default="rex_dc_results",
        help="Output directory for adjacency and graph files.",
    )
    parser.add_argument(
        "--superstructure",
        default="rex",
        choices=["rex", "pc"],
        help="Choose the superstructure learner: 'rex' (dense bootstrapped union) or 'pc'.",
    )
    parser.add_argument("--bootstraps", type=int, default=30)
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument(
        "--pc-alpha",
        type=float,
        default=0.2,
        help="Significance threshold for PC when using --superstructure pc.",
    )
    parser.add_argument(
        "--pc-min-degree",
        type=int,
        default=1,
        help="Minimum degree enforced via MI densification for PC superstructures.",
    )
    parser.add_argument(
        "--pc-mi-neighbors",
        type=int,
        default=3,
        help="k-NN parameter for mutual-information estimates used during PC densification.",
    )
    parser.add_argument(
        "--partition",
        default="edge_cover",
        choices=["edge_cover", "expansive_causal", "modularity"],
    )
    parser.add_argument(
        "--local",
        default="RFCI-PAG",
        choices=["RFCI-PAG", "PC", "GES", "NOTEARS", "REX-SKEL"],
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=20,
        help="Approximate target size of each partition subset.",
    )
    parser.add_argument(
        "--sampling-split",
        type=float,
        default=0.66,
        help="Fraction of samples used in each REX bootstrap iteration.",
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=0,
        help="Number of parallel jobs for REX bootstrapping (0/1 = sequential).",
    )
    parser.add_argument(
        "--auto-params",
        action="store_true",
        help="Auto-tune REX and partition parameters with a lightweight pilot run.",
    )
    parser.add_argument(
        "--max-ram-frac",
        type=float,
        default=0.35,
        help="Maximum fraction of available RAM a single SHAP iteration may consume when auto-tuning.",
    )
    parser.add_argument(
        "--pilot-bootstraps",
        type=int,
        default=3,
        help="Number of pilot bootstraps used during auto-tuning.",
    )
    parser.add_argument(
        "--nn-accelerator",
        default="cpu",
        choices=["cpu", "auto", "gpu", "cuda", "mps"],
        help="Accelerator to train the NN regressor ('gpu'/'cuda' uses CUDA, 'auto' picks automatically).",
    )
    parser.add_argument(
        "--local-rex-bootstraps",
        type=int,
        default=12,
        help="Bootstraps for local REX when --local REX-SKEL is selected.",
    )
    parser.add_argument(
        "--local-rex-tau",
        type=float,
        default=0.10,
        help="Selection threshold for local REX when --local REX-SKEL is selected.",
    )
    parser.add_argument(
        "--local-rex-models",
        default="gbt,nn",
        help="Comma-separated list of regressors for local REX (subset of {'gbt','nn'}).",
    )
    parser.add_argument(
        "--local-rex-sampling-split",
        type=float,
        default=0.66,
        help="Sampling split for local REX bootstrapping.",
    )
    parser.add_argument(
        "--local-rex-parallel-jobs",
        type=int,
        default=0,
        help="Parallel jobs for local REX bootstrapping.",
    )
    parser.add_argument(
        "--local-rex-nn-accelerator",
        default="auto",
        choices=["cpu", "auto", "gpu", "cuda", "mps"],
        help="Accelerator for the NN regressor in local REX (if selected).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logs from REX.",
    )

    args = parser.parse_args()
    dataset = pd.read_csv(args.csv)
    superstructure_mode = args.superstructure.lower()

    local_rex_kwargs: Optional[Dict[str, object]] = None
    if args.local == "REX-SKEL":
        raw_models = [
            token.strip().lower()
            for token in args.local_rex_models.split(",")
            if token.strip()
        ]
        if not raw_models:
            raise ValueError(
                "At least one regressor must be provided via --local-rex-models when using --local REX-SKEL."
            )
        valid_models = {"gbt", "nn"}
        ordered_models: List[str] = []
        seen = set()
        for model in raw_models:
            if model not in valid_models:
                raise ValueError(
                    f"Unsupported local REX regressor '{model}'. "
                    "Use a subset of {'gbt', 'nn'}."
                )
            if model not in seen:
                ordered_models.append(model)
                seen.add(model)
        local_rex_kwargs = {
            "bootstraps": max(1, args.local_rex_bootstraps),
            "tau": args.local_rex_tau,
            "models": tuple(ordered_models),
            "sampling_split": args.local_rex_sampling_split,
            "parallel_jobs": args.local_rex_parallel_jobs,
            "nn_accelerator": args.local_rex_nn_accelerator,
            "verbose": args.verbose,
        }

    prefix = "rex"
    if superstructure_mode == "pc":
        if args.auto_params:
            print(
                "[info] --auto-params is currently ignored for the PC superstructure "
                "pipeline."
            )
        result = pc_then_rex_divide_and_conquer(
            dataset,
            pc_alpha=args.pc_alpha,
            ensure_min_degree=max(0, args.pc_min_degree),
            mi_neighbors=max(1, args.pc_mi_neighbors),
            partition_algo=args.partition,
            target_subset_size=args.subset,
            local=args.local,
            local_rex_kwargs=local_rex_kwargs,
            finite_sample_screen=True,
        )
        prefix = "pc"
    else:
        bootstraps = args.bootstraps
        tau = args.tau
        sampling_split = args.sampling_split
        parallel_jobs = args.parallel_jobs
        nn_accelerator = args.nn_accelerator
        target_subset_size = args.subset

        if args.auto_params:
            ram_frac = max(0.05, min(args.max_ram_frac, 0.9))
            auto = auto_select_params(
                dataset,
                max_ram_frac=ram_frac,
                pilot_bootstraps=max(1, args.pilot_bootstraps),
                nn_accelerator_hint=args.nn_accelerator,
            )
            bootstraps = auto.bootstraps
            tau = auto.tau
            sampling_split = auto.sampling_split
            parallel_jobs = auto.parallel_jobs
            nn_accelerator = auto.nn_accelerator
            target_subset_size = auto.target_subset_size
            if args.verbose:
                print(
                    "[auto] "
                    f"bootstraps={bootstraps} "
                    f"tau={tau:.2f} "
                    f"sampling_split={sampling_split:.2f} "
                    f"avg_degree≈{auto.avg_degree:.1f} "
                    f"subset≈{target_subset_size} "
                    f"parallel_jobs={parallel_jobs} "
                    f"nn_accel={nn_accelerator}"
                )

        result = rex_divide_and_conquer(
            dataset,
            bootstraps=bootstraps,
            tau=tau,
            partition_algo=args.partition,
            local=args.local,
            target_subset_size=target_subset_size,
            sampling_split=sampling_split,
            parallel_jobs=parallel_jobs,
            nn_accelerator=nn_accelerator,
            verbose=args.verbose,
            local_rex_kwargs=local_rex_kwargs,
        )

    _write_outputs(result, args.out, superstructure_prefix=prefix)
    print(f"Pipeline finished. Results stored in: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
