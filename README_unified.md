# obsidian Unified APO Server

A local web server that combines **obsidian**'s Dash UI with multi-optimizer support and benchmarking.
Run four different Bayesian Optimisation backends side-by-side and compare their convergence on standard test functions.

---

## Quick start

```bash
# Install obsidian and optional backends (see Dependencies below)
pip install -e .

# Launch the server
python app_unified.py
# → open http://127.0.0.1:8050
```

---

## What's new vs the standard `app.py`

All original obsidian tabs are preserved unchanged:

| Tab | Description |
|---|---|
| Data | Upload / edit experimental data |
| Config | Parameter space and surrogate settings |
| Optimize | Fit surrogate, suggest next experiments |
| Plots | Response surface and SHAP plots |
| Predict | Forward prediction on new inputs |

One new tab is added:

| Tab | Description |
|---|---|
| **Benchmark** | Compare multiple optimizer backends on standard test functions |

---

## Benchmark tab

### Controls

| Control | Options | Default |
|---|---|---|
| Test function | shifted_parabola, branin, rosenbrock, hartmann3, ackley2d | shifted_parabola |
| Optimizers | Obsidian, BoFire, BayBe, EDBO+ (greyed-out if not installed) | all available |
| Initial points | 3 – 20 | 5 |
| BO iterations | 5 – 50 | 20 |
| Random seed | any integer | 0 |

### Output

- **Convergence plot** — best value found vs cumulative evaluations, one line per optimizer, with a dashed reference line at the known global optimum.
- **Summary table** — best found, total evaluations, gap to optimum for each optimizer.

---

## Supported optimizers

| Name | Backend | Install | Notes |
|---|---|---|---|
| Obsidian | `obsidian.optimizer.BayesianOptimizer` | included | Always available |
| BoFire | `bofire.strategies.SoboStrategy` | `pip install bofire` | Single-objective BO |
| BayBe | `baybe.recommenders.BotorchRecommender` | `pip install baybe` | Stateless recommend API |
| EDBO+ | `edbo.plus.optimizer_botorch.EDBOplus` | `pip install edbo` | Discrete-scope mode |

The UI automatically detects which backends are installed and disables the toggles for missing ones.

### EDBO+ note

EDBOplus is designed for reaction optimisation over a **pre-defined discrete scope** (a list of all possible experiments). The wrapper here automatically materialises a fine grid over the continuous parameter space (30 points per dimension for ≤ 2D, 1 000 LHS points for higher dimensions) to serve as the scope.

---

## Benchmark test functions

| Key | Dimensions | Global optimum | Description |
|---|---|---|---|
| `shifted_parabola` | 1 | 0.0 at x₁ = 0.5 | Simple convex baseline |
| `branin` | 2 | −0.3979 (three locations) | Classic BO benchmark |
| `rosenbrock` | 2 | 0.0 at (1, 1) | Banana-shaped valley |
| `hartmann3` | 3 | 3.8628 | Multi-local-optima |
| `ackley2d` | 2 | 0.0 at origin | Highly multimodal |

All functions are **maximised** (minimisation functions are negated so the UI convention is consistent).

---

## Architecture

```
obsidian/
└── unified/
    ├── __init__.py            # package exports
    ├── optimizer_wrappers.py  # BaseOptimizerWrapper + four adapters
    ├── benchmarks.py          # BenchmarkFunction dataclass + registry
    ├── benchmark_runner.py    # run_benchmark / run_comparison loop
    └── dash_benchmark.py      # Dash "Benchmark" tab + callbacks
app_unified.py                 # entry point
```

### Optimizer wrapper interface

Every backend is wrapped behind a common five-method interface:

```python
wrapper.setup(param_bounds, minimize)   # configure parameter space
wrapper.initialize(n_init, seed)        # generate initial DoE → DataFrame
wrapper.fit(X, y)                       # train surrogate
wrapper.suggest(n)                      # propose next point(s) → DataFrame
wrapper.is_available()                  # True if backend is importable
```

### Running a benchmark programmatically

```python
from obsidian.unified import (
    ObsidianWrapper, BofireWrapper, BaybeWrapper,
    get_benchmark, run_comparison,
)

results = run_comparison(
    optimizers=[ObsidianWrapper(), BofireWrapper(), BaybeWrapper()],
    benchmark=get_benchmark("branin"),
    n_init=5,
    n_iterations=25,
    seed=42,
)

for r in results:
    best = max(r["best_per_eval"])
    print(f"{r['optimizer']:30s}  best = {best:.4f}")
```

### Adding a new optimizer

1. Subclass `BaseOptimizerWrapper` in `obsidian/unified/optimizer_wrappers.py`.
2. Implement the five abstract methods.
3. Append your class to the `ALL_WRAPPERS` list at the bottom of the file.

The UI will pick it up automatically on the next server start.

### Adding a new benchmark function

1. Write a function `(X: pd.DataFrame) -> pd.Series` in `obsidian/unified/benchmarks.py`.
2. Add a `BenchmarkFunction` entry to `BENCHMARK_REGISTRY`.

---

## Dependencies

### Required (obsidian core)

```
dash
dash-bootstrap-components
plotly
pandas
numpy
torch
botorch
```

### Optional backends

```bash
# BoFire
pip install bofire

# BayBe
pip install baybe

# EDBO+
pip install edbo          # or install from the edboplus repo
```

---

## File reference

| File | Role |
|---|---|
| `app_unified.py` | Entry point — run this to start the server |
| `app.py` | Original obsidian server (unchanged) |
| `obsidian/unified/optimizer_wrappers.py` | Adapter classes for each BO backend |
| `obsidian/unified/benchmarks.py` | Test functions and registry |
| `obsidian/unified/benchmark_runner.py` | `run_benchmark` / `run_comparison` |
| `obsidian/unified/dash_benchmark.py` | Dash tab layout and callbacks |
