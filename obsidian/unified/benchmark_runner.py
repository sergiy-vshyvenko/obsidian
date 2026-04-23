"""Run single or comparative optimizer benchmarks.

Usage example::

    from obsidian.unified import ObsidianWrapper, get_benchmark, run_comparison

    results = run_comparison(
        optimizers=[ObsidianWrapper()],
        benchmark=get_benchmark("branin"),
        n_init=5,
        n_iterations=20,
    )
    # results is a list of dicts, one per optimizer
    # each dict has keys: optimizer, best_per_eval, n_evals, X_all, y_all
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from obsidian.unified.optimizer_wrappers import BaseOptimizerWrapper
from obsidian.unified.benchmarks import BenchmarkFunction


def run_benchmark(
    optimizer: BaseOptimizerWrapper,
    benchmark: BenchmarkFunction,
    n_init: int = 5,
    n_iterations: int = 20,
    n_suggest: int = 1,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run a single optimizer on a benchmark function.

    Parameters
    ----------
    optimizer:
        An **unsetup** wrapper instance (``setup`` is called here).
    benchmark:
        A ``BenchmarkFunction`` from ``BENCHMARK_REGISTRY``.
    n_init:
        Number of initial DoE points.
    n_iterations:
        Number of BO iterations (fit-suggest cycles).
    n_suggest:
        Number of candidates to suggest per iteration.
    seed:
        Random seed for initialisation.

    Returns
    -------
    dict with keys:
        ``optimizer``     – human-readable optimizer label (str)
        ``best_per_eval`` – best objective seen after each evaluation (list[float])
        ``n_evals``       – cumulative evaluation index (list[int])
        ``X_all``         – all evaluated inputs (pd.DataFrame)
        ``y_all``         – all evaluated outputs (pd.Series)
    """
    optimizer.setup(benchmark.param_bounds, minimize=benchmark.minimize)

    X_init = optimizer.initialize(n_init, seed=seed)
    y_init = benchmark(X_init)

    X_all = X_init.copy().reset_index(drop=True)
    y_all = y_init.copy().reset_index(drop=True)

    def _best(series: pd.Series) -> float:
        return float(series.min() if benchmark.minimize else series.max())

    # Record best after each initial evaluation
    best_per_eval: List[float] = [
        _best(y_all.iloc[: i + 1]) for i in range(len(y_all))
    ]

    for _iter in range(n_iterations):
        try:
            optimizer.fit(X_all, y_all)
            X_new = optimizer.suggest(n_suggest)
            if X_new is None or X_new.empty:
                break
        except Exception as exc:  # noqa: BLE001
            print(f"[{optimizer.name}] iteration {_iter} failed: {exc}")
            break

        y_new = benchmark(X_new)
        X_all = pd.concat([X_all, X_new], ignore_index=True)
        y_all = pd.concat([y_all, y_new], ignore_index=True)

        for _ in range(len(y_new)):
            best_per_eval.append(_best(y_all))

    return {
        "optimizer": optimizer.label,
        "best_per_eval": best_per_eval,
        "n_evals": list(range(1, len(best_per_eval) + 1)),
        "X_all": X_all,
        "y_all": y_all,
    }


def run_comparison(
    optimizers: List[BaseOptimizerWrapper],
    benchmark: BenchmarkFunction,
    n_init: int = 5,
    n_iterations: int = 20,
    n_suggest: int = 1,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Run multiple optimizers on the same benchmark and collect all results.

    Each optimizer receives the same ``seed`` so initial points are
    comparably positioned (where the wrapper supports deterministic init).
    """
    return [
        run_benchmark(
            optimizer=opt,
            benchmark=benchmark,
            n_init=n_init,
            n_iterations=n_iterations,
            n_suggest=n_suggest,
            seed=seed,
        )
        for opt in optimizers
    ]
