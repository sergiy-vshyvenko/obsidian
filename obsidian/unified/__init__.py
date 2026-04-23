"""Unified optimizer interface and benchmarking for obsidian, BoFire, BayBe, and EDBO+."""

from obsidian.unified.optimizer_wrappers import (
    ObsidianWrapper,
    BofireWrapper,
    BaybeWrapper,
    EdboplusWrapper,
    get_available_optimizers,
    ALL_WRAPPERS,
)
from obsidian.unified.benchmarks import BENCHMARK_REGISTRY, get_benchmark
from obsidian.unified.benchmark_runner import run_benchmark, run_comparison

__all__ = [
    "ObsidianWrapper",
    "BofireWrapper",
    "BaybeWrapper",
    "EdboplusWrapper",
    "get_available_optimizers",
    "ALL_WRAPPERS",
    "BENCHMARK_REGISTRY",
    "get_benchmark",
    "run_benchmark",
    "run_comparison",
]
