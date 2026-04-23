"""Standard BO benchmark test functions.

All functions accept a ``pd.DataFrame`` with named columns and return a
``pd.Series`` named ``"y"``.  Convention: **maximise** the returned value
(functions that are naturally minimised are negated so a single
``minimize=False`` default works across all wrappers).

The ``BENCHMARK_REGISTRY`` dict maps a short string key to a
``BenchmarkFunction`` dataclass that also stores:

* ``param_bounds`` – ``{name: (min, max)}`` dict
* ``minimize``     – ``False`` for all functions here (negated maxima)
* ``optimum``      – known global optimum of the **returned** ``y`` value
* ``description``  – one-liner shown in the UI
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

ParamBounds = Dict[str, Tuple[float, float]]


@dataclass
class BenchmarkFunction:
    """Container for a benchmark test function and its metadata."""

    name: str
    func: Callable[[pd.DataFrame], pd.Series]
    param_bounds: ParamBounds
    minimize: bool
    optimum: Optional[float]
    description: str

    def __call__(self, X: pd.DataFrame) -> pd.Series:
        return self.func(X)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def _shifted_parabola(X: pd.DataFrame) -> pd.Series:
    """Shifted parabola. Max = 0 at x1 = 0.5."""
    x = X.iloc[:, 0].values
    y = -4.0 * (x - 0.5) ** 2
    return pd.Series(y, name="y")


def _branin(X: pd.DataFrame) -> pd.Series:
    """Branin (negated). Three global maxima ≈ -0.3979 at known locations."""
    x1 = X["x1"].values
    x2 = X["x2"].values
    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)
    y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return pd.Series(-y, name="y")  # negate -> maximise


def _rosenbrock(X: pd.DataFrame) -> pd.Series:
    """Rosenbrock (negated). Global max = 0 at (1, 1)."""
    x1 = X["x1"].values
    x2 = X["x2"].values
    y = 100.0 * (x2 - x1 ** 2) ** 2 + (x1 - 1.0) ** 2
    return pd.Series(-y, name="y")  # negate -> maximise


def _hartmann3(X: pd.DataFrame) -> pd.Series:
    """Hartmann 3D (negated). Global max ≈ 3.8628 at known location."""
    A = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ])
    P = 1e-4 * np.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [381, 5743, 8828],
    ])
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    Xv = X[["x1", "x2", "x3"]].values
    y = np.zeros(len(Xv))
    for i, xi in enumerate(Xv):
        inner = np.sum(A * (xi - P) ** 2, axis=1)
        y[i] = -np.sum(alpha * np.exp(-inner))
    return pd.Series(-y, name="y")  # negate -> maximise


def _ackley2d(X: pd.DataFrame) -> pd.Series:
    """Ackley 2D (negated, multimodal). Global max = 0 at origin."""
    x1 = X["x1"].values
    x2 = X["x2"].values
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    sq = np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))
    cos_sum = 0.5 * (np.cos(c * x1) + np.cos(c * x2))
    y = -a * np.exp(-b * sq) - np.exp(cos_sum) + a + np.e
    return pd.Series(-y, name="y")  # negate -> maximise


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY: Dict[str, BenchmarkFunction] = {
    "shifted_parabola": BenchmarkFunction(
        name="shifted_parabola",
        func=_shifted_parabola,
        param_bounds={"x1": (0.0, 1.0)},
        minimize=False,
        optimum=0.0,
        description="Shifted parabola (1D). Global max = 0 at x1 = 0.5.",
    ),
    "branin": BenchmarkFunction(
        name="branin",
        func=_branin,
        param_bounds={"x1": (-5.0, 10.0), "x2": (0.0, 15.0)},
        minimize=False,
        optimum=-0.397887,
        description="Branin (2D). Three global maxima ≈ -0.398.",
    ),
    "rosenbrock": BenchmarkFunction(
        name="rosenbrock",
        func=_rosenbrock,
        param_bounds={"x1": (-2.0, 2.0), "x2": (-1.0, 3.0)},
        minimize=False,
        optimum=0.0,
        description="Rosenbrock (2D). Global max = 0 at (1, 1).",
    ),
    "hartmann3": BenchmarkFunction(
        name="hartmann3",
        func=_hartmann3,
        param_bounds={"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)},
        minimize=False,
        optimum=3.8628,
        description="Hartmann 3D. Global max ≈ 3.863.",
    ),
    "ackley2d": BenchmarkFunction(
        name="ackley2d",
        func=_ackley2d,
        param_bounds={"x1": (-5.0, 5.0), "x2": (-5.0, 5.0)},
        minimize=False,
        optimum=0.0,
        description="Ackley 2D (multimodal). Global max = 0 at origin.",
    ),
}


def get_benchmark(name: str) -> BenchmarkFunction:
    """Retrieve a benchmark by key, raising KeyError if not found."""
    if name not in BENCHMARK_REGISTRY:
        raise KeyError(
            f"Unknown benchmark '{name}'. "
            f"Available: {list(BENCHMARK_REGISTRY.keys())}"
        )
    return BENCHMARK_REGISTRY[name]
