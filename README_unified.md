# obsidian Unified APO Server

A local web server that combines **obsidian**'s Dash UI with multi-optimizer support and benchmarking.
Run Bayesian Optimisation backends side-by-side and compare their convergence on standard test functions.

---

## Quick start

```bash
# Clone and install obsidian with the Dash app extras
git clone <repo>
cd obsidian
pip install -e ".[app]"

# Launch the server
python app_unified.py
# → open http://127.0.0.1:8050
```

> **Note:** plain `pip install -e .` installs the core library only.
> The Dash UI (and Pillow for the logo) live in the `[app]` extra.

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
| Obsidian | `obsidian.optimizer.BayesianOptimizer` | `pip install -e ".[app]"` | Always available |
| BoFire | `bofire.strategies.SoboStrategy` | `pip install "bofire[optimization]"` | Compatible with BayBe |
| BayBe | `baybe.recommenders.BotorchRecommender` | `pip install baybe` | Compatible with BoFire |
| EDBO+ | `edbo.plus.optimizer_botorch.EDBOplus` | see note below | **Separate env required** |

The UI automatically detects which backends are installed and disables toggles for missing ones.

---

## Dependency compatibility

### Full matrix

| Package | Obsidian | BoFire | BayBe | EDBO+ |
|---|---|---|---|---|
| `botorch` | `>=0.11.1,<1` ✅ | `>=0.16.1` ✅ | `>=0.13.0,<1` ✅ | `==0.5.0` ❌ |
| `gpytorch` | `^1.11` | via botorch | `>=1.9.1,<2` ✅ | `==1.5.1` ❌ |
| `torch` | `>=2.3.0,<3` | via botorch | `>=1.13.1,<3` ✅ | `==1.10.0` ❌ |
| `numpy` | `^1.26` | unconstrained | `>=1.24.1,<3` ✅ | `==1.21.5` ❌ |
| `pandas` | `^2.2.2` | unconstrained | `>=1.4.2,<3` ✅ | `==1.3.4` ❌ |
| `scipy` | `^1.13.1` | `>=1.7` ✅ | `>=1.10.1` ✅ | `>=1.10.0` ✅ |
| `scikit-learn` | `^1.5.1` (opt) | `>=1.0.0` ✅ | `>=1.1.1,<2` ✅ | `>=1.4` ✅ |
| `dash` | optional `[app]` | — | — | — |
| `pillow` | optional `[app]` | — | — | — |

✅ Compatible &nbsp;&nbsp; ❌ Incompatible (irreconcilable version conflict)

### What changed in obsidian's `pyproject.toml`

| Dep | Before | After | Reason |
|---|---|---|---|
| `botorch` | `^0.11.1` (→ `<0.12`) | `>=0.11.1,<1` | BoFire needs `>=0.16.1`, BayBe needs `>=0.13.0` |
| `torch` | `==2.3.0` (exact pin) | `>=2.3.0,<3` | BayBe allows `<3`; exact pin prevents upgrades |
| `pillow` | missing | `>=9.0` (optional `[app]`) | `app.py` and `app_unified.py` use `PIL.Image` |

### Environments

**Obsidian + BoFire + BayBe (recommended for multi-optimizer benchmarking):**
```bash
conda create -n boopt python=3.11
conda activate boopt
pip install -e ".[app]"
pip install "bofire[optimization]" baybe
python app_unified.py
```

**Obsidian only (no optional backends):**
```bash
pip install -e ".[app]"
python app_unified.py   # only Obsidian optimizer available
```

**EDBO+ (isolated — cannot share env with BoFire or BayBe):**
```bash
conda create -n edboplus python=3.10
conda activate edboplus
pip install edbo
pip install -e ".[app]"
python app_unified.py   # only Obsidian + EDBO+ available
```

### Why EDBO+ must be isolated

EDBO+ pins `botorch==0.5.0` (2021), `torch==1.10.0`, `numpy==1.21.5`, and
`pandas==1.3.4` — all irreconcilably older than what Obsidian, BoFire, and BayBe
require. It cannot share a pip environment with any of the other three.

The EDBO+ wrapper also materialises a fine grid over the continuous parameter space
(30 points/dim for ≤2D, 1 000 LHS points otherwise) as its discrete scope, which is
how EDBO+ is designed to work in reaction-optimisation workflows.

---

## Benchmark test functions

| Key | Dimensions | Global optimum | Description |
|---|---|---|---|
| `shifted_parabola` | 1 | 0.0 at x₁ = 0.5 | Simple convex baseline |
| `branin` | 2 | −0.3979 (three locations) | Classic BO benchmark |
| `rosenbrock` | 2 | 0.0 at (1, 1) | Banana-shaped valley |
| `hartmann3` | 3 | 3.8628 | Multi-local-optima |
| `ackley2d` | 2 | 0.0 at origin | Highly multimodal |

All functions are **maximised** (minimisation functions are negated for consistent UI convention).

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

```python
wrapper.setup(param_bounds, minimize)   # configure parameter space
wrapper.initialize(n_init, seed)        # generate initial DoE → DataFrame
wrapper.fit(X, y)                       # train surrogate
wrapper.suggest(n)                      # propose next point(s) → DataFrame
wrapper.is_available()                  # True if backend is importable
```

### Running benchmarks programmatically

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
3. Append your class to `ALL_WRAPPERS` at the bottom of the file.

### Adding a new benchmark function

1. Write `(X: pd.DataFrame) -> pd.Series` in `obsidian/unified/benchmarks.py`.
2. Add a `BenchmarkFunction` entry to `BENCHMARK_REGISTRY`.

---

## File reference

| File | Role |
|---|---|
| `app_unified.py` | Entry point — `python app_unified.py` |
| `app.py` | Original obsidian server (unchanged) |
| `pyproject.toml` | Dependency spec (botorch relaxed, pillow added) |
| `obsidian/unified/optimizer_wrappers.py` | Adapter classes |
| `obsidian/unified/benchmarks.py` | Test functions and registry |
| `obsidian/unified/benchmark_runner.py` | `run_benchmark` / `run_comparison` |
| `obsidian/unified/dash_benchmark.py` | Dash tab layout and callbacks |
| `README_unified.md` | This file |
