# obsidian Unified APO Server

A local web server that combines **obsidian**'s Dash UI with multi-optimizer support and benchmarking.
Run Bayesian Optimisation backends side-by-side and compare their convergence on standard test functions.

---

## Quick start

```bash
# Install obsidian with the Dash UI extras and optional backends (see Dependencies below)
pip install -e ".[app]"

# Launch the server
python app_unified.py
# → open http://127.0.0.1:8050
```

> `[app]` installs `dash`, `dash-daq`, `dash-bootstrap-components`, and `pillow` — all required
> to run the server. Omitting it leaves only the core obsidian library without a UI.

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
| BoFire | `bofire.strategies.SoboStrategy` | `pip install bofire[optimization]` | Compatible with BayBe |
| BayBe | `baybe.recommenders.BotorchRecommender` | `pip install baybe` | Compatible with BoFire |
| EDBO+ | `edbo.plus.optimizer_botorch.EDBOplus` | see note below | **Separate env required** |

The UI automatically detects which backends are installed and disables toggles for missing ones.

### EDBO+ — isolation required

EDBO+ pins very old versions of core packages (`botorch==0.5.0`, `torch==1.10.0`,
`numpy==1.21.5`, `pandas==1.3.4`) that **directly conflict** with BoFire and BayBe.
It cannot be installed in the same environment as either of those two libraries.

**Recommended approach: use a dedicated conda environment for EDBO+.**

```bash
conda create -n edboplus python=3.10
conda activate edboplus
pip install edbo             # installs its pinned deps
pip install -e ".[app]"      # install obsidian with Dash UI
python app_unified.py        # only Obsidian + EDBO+ will be available
```

For the combined Obsidian + BoFire + BayBe environment (EDBO+ excluded):

```bash
conda create -n boopt python=3.11
conda activate boopt
pip install -e ".[app]"
pip install bofire[optimization] baybe
python app_unified.py
```

### EDBO+ scope note

EDBOplus is designed for reaction optimisation over a **pre-defined discrete scope**.
The wrapper here automatically materialises a fine grid over the continuous parameter
space (30 points per dimension for ≤ 2D, 1 000 LHS points otherwise) to serve as the
scope, matching how EDBO+ is typically used in practice.

---

## Dependency compatibility matrix

| Package | BoFire | BayBe | EDBO+ |
|---|---|---|---|
| `botorch` | `>=0.16.1` | `>=0.13.0,<1` | `==0.5.0` ❌ |
| `gpytorch` | via botorch | `>=1.9.1,<2` | `==1.5.1` ❌ |
| `torch` | via botorch | `>=1.13.1,<3` | `==1.10.0` ❌ |
| `numpy` | unconstrained | `>=1.24.1,<3` | `==1.21.5` ❌ |
| `pandas` | unconstrained | `>=1.4.2,<3` | `==1.3.4` ❌ |
| `scipy` | `>=1.7` | `>=1.10.1` | `>=1.10.0` ✅ |
| `scikit-learn` | `>=1.0.0` | `>=1.1.1,<2` | `>=1.4` ✅ |
| `sympy` | `>=1.12` | — | `==1.9` ❌ |
| `pymoo` | `>=0.6.0` | — | `==0.5.0` ❌ |
| Python | `>=3.10,<3.15` | `>=3.10,<3.15` | 3.9–3.11 ✅ |

✅ Compatible &nbsp;&nbsp; ❌ Incompatible (irreconcilable version conflict)

**BoFire and BayBe are mutually compatible** and can share one environment.  
**EDBO+ requires its own isolated environment** due to ancient pinned versions of
`botorch`, `torch`, `numpy`, and `pandas`.

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
wrapper.setup(param_bounds, param_categories, objectives, minimize)  # configure space + objectives
wrapper.initialize(n_init, seed)   # generate initial DoE → DataFrame
wrapper.fit(X, y)                  # train surrogate (y: Series for SOO, DataFrame for MOO)
wrapper.suggest(n)                 # propose next point(s) → DataFrame
wrapper.predict(X)                 # posterior mean + std → DataFrame (BayBe / Obsidian only)
wrapper.is_available()             # True if backend is importable
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

The UI picks it up automatically on the next server start.

### Adding a new benchmark function

1. Write a function `(X: pd.DataFrame) -> pd.Series` in `obsidian/unified/benchmarks.py`.
2. Add a `BenchmarkFunction` entry to `BENCHMARK_REGISTRY`.

---

## Installation recipes

Clone the repo first:

```bash
git clone https://github.com/sergiy-vshyvenko/obsidian.git
cd obsidian
```

### Obsidian + BoFire + BayBe (recommended)

Verified working on Python 3.11:

```bash
conda create -n boopt python=3.11
conda activate boopt
pip install -e ".[app]"
pip install "bofire[optimization]" "cvxpy==1.5.4" baybe
```

> **Why `cvxpy==1.5.4`?** BoFire pulls in cvxpy, and cvxpy ≥ 1.6 requires NumPy 2.x.
> Pinning to 1.5.4 keeps NumPy on the 1.26.x series that obsidian's core requires.

Verified package versions (as of April 2026):

| Package | Version |
|---|---|
| Python | 3.11 |
| torch | 2.3.0 |
| botorch | 0.11.3 |
| gpytorch | 1.12 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scipy | 1.13.1 |
| dash | 4.1.0 |
| dash-bootstrap-components | 2.0.4 |
| dash-daq | 0.6.0 |
| pillow | 12.1.1 |
| cvxpy | 1.5.4 |

### Obsidian + EDBO+ (isolated)

EDBO+ pins `botorch==0.5.0` / `torch==1.10.0` / `numpy==1.21.5` — incompatible with BoFire/BayBe.
Use a dedicated environment:

```bash
conda create -n edboplus python=3.10
conda activate edboplus
pip install edbo
pip install -e ".[app]"
```

### Obsidian only (no optional backends)

```bash
conda create -n obsidian python=3.11
conda activate obsidian
pip install -e ".[app]"
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
