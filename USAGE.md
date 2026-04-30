# Using the obsidian Unified APO Server

This guide walks through every tab of `app_unified.py` — what each control does,
how to read the outputs, and practical tips for getting useful results.

---

## Starting the server

```bash
python app_unified.py
```

Open your browser to **http://127.0.0.1:8050**.

The server reloads automatically when you change Python files (debug mode is on by default).
Press `Ctrl+C` in the terminal to stop it.

---

## Interface overview

The app has six tabs:

| Tab | What it is for |
|---|---|
| **Data** | Load your data and define response objectives |
| **Config** | Define the parameter space and surrogate model |
| **Optimize** | Choose a backend, fit the surrogate, and get suggestions |
| **Plots** | Visualise the response surface and feature importance |
| **Predict** | Get a prediction at any point in the space |
| **Benchmark** | Compare optimizer backends on standard test functions |

---

## Data tab

### What it shows

A table of your experimental data — input parameters (X) and measured response(s) (y).
When you start the server for the first time a built-in example dataset is loaded:

| Parameter | Type | Range |
|---|---|---|
| Temperature | Continuous | −10 to 30 |
| Concentration | Continuous | 10 to 150 |
| Enzyme | Continuous | 0.01 to 0.30 |
| Variant | Categorical | MRK001, MRK002, MRK003 |
| Stir Rate | Ordinal | Low, Medium, High |

The output column is **Yield**, simulated from a shifted parabola with 5 % noise.

### Loading your own data

1. Click **Upload CSV** and select a file.
2. The first row must be a header. Column names must match your parameter names exactly.
3. Response column(s) can be anywhere in the file — you select them below the table.

### Response selection (single-objective)

The **Response Selection** card beneath the table lets you pick which column is the
optimization target.

1. Choose the column from the **Data Column** dropdown.
2. Use the **Maximize / Minimize** toggle to set the optimization direction.

### Multi-objective optimization (MOO)

To optimize two or more responses simultaneously:

1. Set up the primary response as above.
2. Click **Add Response** to add a second objective row. Each row has:
   - A column dropdown — pick any numeric column from your data.
   - A **Maximize / Minimize** toggle.
   - A **Delete** button to remove it.
3. Repeat for a third objective if needed.

All selected response columns are automatically excluded from the input parameter cards below.

> **Which backends support MOO?**
> - **Obsidian** — full MOO support via NEHVI (noisy expected hypervolume improvement). Supports any parameter type.
> - **BoFire** — MOO via `qNEHVIStrategy`. Continuous parameters only.
> - **BayBe** — MOO via `ParetoObjective`. Continuous parameters only.

### Editing data

Click any cell in the table to edit it directly. Rows can be added or deleted.
Changes stay in memory for the current session — they are not written back to your file.

---

## Config tab

### Parameter space

Defines the boundaries and types of each input variable.

- **Continuous** — specify min and max (e.g. Temperature: −10, 30).
- **Categorical** — comma-separated list of options (e.g. Variant: MRK001, MRK002).
- **Ordinal** — ordered list, treated as equally spaced integers internally.

Changes here reset the surrogate — go back to the Optimize tab to refit.

### Surrogate model settings

Controls the Gaussian Process model used by obsidian:

| Setting | What it does |
|---|---|
| Kernel | Covariance function (RBF, Matern 5/2, etc.) |
| ARD | Automatic Relevance Determination — separate length scales per parameter |
| Noise | Expected observation noise level |

Leave these at defaults until you have a reason to change them.

---

## Optimize tab

### Step 0 — Choose a backend

The **Optimizer Backend** dropdown at the top of the tab selects which library does the fitting
and suggestion. Only backends installed in your current Python environment appear.

| Backend | Library | Install |
|---|---|---|
| Obsidian (BoTorch) | obsidian (this package) | always available |
| BoFire (BoTorch) | bofire | `pip install "bofire[optimization]"` |
| BayBe (BoTorch) | baybe | `pip install baybe` |

If you change the backend after fitting, the app warns you to refit before suggesting.

### Step 1 — Fit the surrogate

Click **Fit** to train the model on your current data.

- **Obsidian** fits a Gaussian Process and reports R² and marginal log-likelihood per response.
- **BoFire / BayBe** fit their internal surrogates and report a confirmation card (no R² exposed).

For MOO (multiple responses selected in the Data tab), each response gets its own surrogate.
The Regression Statistics card shows R² and MLL for each one individually.

### Step 2 — Suggest next experiments

Click **Optimize** to generate candidate experiments.

The table shows the recommended parameter settings. For **Obsidian** it also shows predicted
response values and acquisition scores. For **BoFire / BayBe** it shows parameter columns only
(response predictions are internal to those libraries).

**Acquisition function notes:**
- For single-objective, the acquisition function set in the Config tab is used (EI, PI, UCB, …).
- For multi-objective (2+ responses), NEHVI is used automatically regardless of the Config tab setting.

Copy the suggested values into your lab notebook or process system, run the experiment,
add the result to the Data tab, refit, and repeat.

### Download

Click **Download Suggested Candidates** to save the suggestions as a CSV.

---

## Plots tab

Available after fitting the surrogate (Obsidian backend only).

### Response surface

A 2D contour or 3D surface plot of the predicted mean across two parameters.
Use the dropdowns to choose which two parameters to show on X and Y axes.
All other parameters are held at their current best-guess value.

**Reading the plot:**
- Darker / higher regions = predicted higher yield.
- The uncertainty band (shown as a ribbon on 1D plots) widens where data is sparse.
- Points on the plot are your actual observations.

### SHAP importance

Bar chart showing which parameters have the most influence on the prediction.
Higher bars = stronger effect. Use this to decide which parameters to vary in future experiments
and which can be held constant.

---

## Predict tab

Enter specific parameter values to get a point prediction with uncertainty.

| Output | Meaning |
|---|---|
| Predicted mean | Best estimate of y at that point |
| Std deviation | 1-sigma uncertainty (68 % credible interval) |
| 95 % interval | [mean − 1.96σ, mean + 1.96σ] |

Useful for sanity-checking the model against known reference points before committing to a suggestion.

---

## Benchmark tab

The Benchmark tab lets you run multiple optimizer backends on the same mathematical
test function and compare their convergence speed side by side.

This does **not** use your experimental data — it evaluates the optimizers on known
mathematical functions so results are reproducible and ground-truth is available.

### Step-by-step walkthrough

#### 1. Choose a test function

| Function | Dims | Global maximum | Character |
|---|---|---|---|
| `shifted_parabola` | 1 | 0.0 at x₁ = 0.5 | Simple convex — use to verify setup |
| `branin` | 2 | −0.398 (3 locations) | Classic BO benchmark |
| `rosenbrock` | 2 | 0.0 at (1, 1) | Banana-shaped narrow valley — hard for surrogates |
| `hartmann3` | 3 | 3.863 | Multiple local optima in 3D |
| `ackley2d` | 2 | 0.0 at origin | Highly multimodal — stress-tests exploration |

**Recommendation:** Start with `shifted_parabola` or `branin` to confirm the setup works,
then use `hartmann3` or `ackley2d` for meaningful comparisons.

All functions are shown as **maximisation** problems (minimisation functions are negated internally).

#### 2. Select optimizers to compare

A toggle switch appears for each available backend.
Backends not installed in your current environment are **greyed out and disabled** automatically.

| Toggle | Backend | Available when |
|---|---|---|
| Obsidian (BoTorch) | obsidian's Campaign / BayesianOptimizer | Always |
| BoFire (BoTorch) | bofire SoboStrategy | `pip install "bofire[optimization]"` |
| BayBe (BoTorch) | baybe BotorchRecommender | `pip install baybe` |
| EDBO+ (BoTorch) | edbo.plus EDBOplus | Separate conda env — see README_unified.md |

Enable at least one optimizer before clicking Run.

#### 3. Set experiment parameters

| Control | What it sets | Typical starting values |
|---|---|---|
| **Initial points** | Number of random / LHS points before BO starts | 5 for 1–2D, 10 for 3D |
| **BO iterations** | Number of BO suggestion-evaluate cycles after init | 20–30 |
| **Random seed** | Makes results reproducible | Any integer (0 is fine) |

#### 4. Click Run Benchmark

A spinner appears while the benchmark runs. Each optimizer goes through:

```
initialize (n_init random points)
  ↓ evaluate test function
for each iteration:
    fit surrogate on all data so far
    suggest 1 new point
    evaluate test function
    record best value seen
```

Run time is roughly `n_optimizers × (n_init + n_iterations)` surrogate fits.
On a laptop: `branin`, 3 optimizers, 5 init + 20 iter ≈ 15–30 seconds.

---

### Reading the results

#### Convergence plot

Each optimizer appears as one coloured line.
The X axis is the total number of function evaluations (init points + BO iterations).
The Y axis is the **best value found so far** at that evaluation count.

- A line that rises quickly = fast convergence (good).
- A flat line = the optimizer is not improving (stuck or exploring).
- The **dashed grey line** is the known global optimum.

#### Summary table

| Column | Meaning |
|---|---|
| Optimizer | Backend name |
| Best Found | Highest y value seen across all evaluations |
| Total Evaluations | n_init + n_iterations (same for all) |
| Gap to Optimum | `|known optimum − best found|` — lower is better |

---

### Tips for meaningful comparisons

**Use the same seed** across runs when comparing — different seeds give different random
initialization. Seed 0 is a good default.

**Run multiple seeds and average** for a fair comparison (use the programmatic API below).

**EDBO+ uses a discrete grid** internally. On continuous benchmarks it searches only pre-sampled
candidate points (up to 1 000 for 3D), so it may underperform. This is expected behaviour.

---

## Programmatic use

### Single-objective benchmark

```python
from obsidian.unified import (
    ObsidianWrapper,
    BofireWrapper,
    BaybeWrapper,
    get_benchmark,
    run_comparison,
)

benchmark = get_benchmark("branin")

results = run_comparison(
    optimizers=[ObsidianWrapper(), BofireWrapper(), BaybeWrapper()],
    benchmark=benchmark,
    n_init=5,
    n_iterations=25,
    seed=42,
)

for r in results:
    best = max(r["best_per_eval"])
    gap  = abs(benchmark.optimum - best)
    print(f"{r['optimizer']:30s}  best={best:.4f}  gap={gap:.4f}")
```

### Multi-objective optimization (programmatic)

Use the `objectives` parameter to pass multiple `(name, minimize)` pairs:

```python
import pandas as pd
from obsidian.unified import BofireWrapper, BaybeWrapper

# Example: maximize yield, minimize cost
objectives = [("yield", False), ("cost", True)]

# BoFire
wrapper = BofireWrapper()
wrapper.setup(param_bounds={"temp": (20, 80), "conc": (0.1, 1.0)}, objectives=objectives)
wrapper.fit(X_train, y_train)   # y_train is a DataFrame with columns ["yield", "cost"]
X_next = wrapper.suggest(n=3)

# BayBe
wrapper = BaybeWrapper()
wrapper.setup(param_bounds={"temp": (20, 80), "conc": (0.1, 1.0)}, objectives=objectives)
wrapper.fit(X_train, y_train)
X_next = wrapper.suggest(n=3)
```

### Multi-seed averaging

```python
import numpy as np
from obsidian.unified import ObsidianWrapper, BofireWrapper, get_benchmark, run_comparison

benchmark = get_benchmark("hartmann3")
gap_by_optimizer = {}

for seed in range(10):
    results = run_comparison(
        optimizers=[ObsidianWrapper(), BofireWrapper()],
        benchmark=benchmark,
        n_init=10,
        n_iterations=30,
        seed=seed,
    )
    for r in results:
        name = r["optimizer"]
        gap  = abs(benchmark.optimum - max(r["best_per_eval"]))
        gap_by_optimizer.setdefault(name, []).append(gap)

for name, gaps in gap_by_optimizer.items():
    print(f"{name:30s}  mean gap = {np.mean(gaps):.4f} ± {np.std(gaps):.4f}")
```

### Available objects

```python
from obsidian.unified import (
    ObsidianWrapper,         # obsidian backend
    BofireWrapper,           # BoFire backend (SOO + MOO)
    BaybeWrapper,            # BayBe backend (SOO + MOO)
    EdboplusWrapper,         # EDBO+ backend (SOO only, separate env)
    ALL_WRAPPERS,            # list of all four classes
    get_available_optimizers,  # returns only importable wrappers
    BENCHMARK_REGISTRY,      # dict of all test functions
    get_benchmark,           # look up by key
    run_benchmark,           # single optimizer run
    run_comparison,          # multiple optimizers on same benchmark
)
```

### `run_benchmark` return format

```python
{
    "optimizer":     "obsidian",           # optimizer name string
    "best_per_eval": [0.1, 0.3, ...],      # best-so-far at each evaluation
    "n_evals":       [1, 2, 3, ...],       # evaluation indices (x axis)
    "X_all":         pd.DataFrame(...),    # all evaluated points
    "y_all":         pd.Series(...),       # all observed y values
}
```

---

## Common issues

### Optimizer backend dropdown shows only one option

Only libraries installed in the current Python environment appear.
Install the optional backends:

```bash
pip install "bofire[optimization]" baybe
```

Then restart the server.

### "Backend changed since last fit — please re-fit the model"

You switched the Optimizer Backend dropdown after clicking Fit. Click **Fit** again with the
new backend selected, then click **Optimize**.

### Fit succeeds but Optimize shows an error for MOO

For multi-objective optimization with Obsidian, the acquisition function is automatically
switched to NEHVI. If you see an error, check that:
- All selected response columns exist in the uploaded data.
- All response columns are numeric.
- You have enough observations (at least 2× the number of responses is recommended).

### "None of the selected optimizers are available" (Benchmark tab)

The backend library is not installed. Install it or switch to the conda environment that has it.

### Benchmark runs but convergence plot is flat

- **n_init** may be too large relative to n_iterations — reduce n_init or increase n_iterations.
- Try `shifted_parabola` (1D) as a sanity check.

### EDBO+ toggle is greyed out

EDBO+ requires an isolated conda environment (see `README_unified.md`).
It cannot be installed alongside BoFire or BayBe.

### Server won't start — `PIL` / `pillow` not found

```bash
pip install pillow
```

Or reinstall with the app extras: `pip install -e ".[app]"`.

### Port 8050 is already in use

Another Dash server is running. Either stop it, or change the port:

```python
# at the bottom of app_unified.py
app.run_server(debug=True, port=8051)
```
