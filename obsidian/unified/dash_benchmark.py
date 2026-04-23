"""Benchmark tab for the unified multi-optimizer comparison server.

Adds a ``\"Benchmark\"`` tab to an existing obsidian Dash app that lets you:

* choose a standard BO test function (Branin, Hartmann-3D, …)
* tick which optimizer backends to compare
* set n_init, n_iterations and random seed
* run all selected optimizers and see a live convergence plot

Integrate with::

    from obsidian.unified.dash_benchmark import setup_benchmark
    setup_benchmark(app, app_tabs)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, html, Output, Input, State

from obsidian.dash.utils import add_tab, center, make_table
from obsidian.unified.benchmarks import BENCHMARK_REGISTRY
from obsidian.unified.optimizer_wrappers import ALL_WRAPPERS


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _benchmark_dropdown_options() -> List[Dict[str, str]]:
    return [
        {"label": f"{k}  —  {v.description}", "value": k}
        for k, v in BENCHMARK_REGISTRY.items()
    ]


def _optimizer_checklist_options() -> List[Dict[str, Any]]:
    return [
        {
            "label": cls.label,
            "value": cls.name,
            "disabled": not cls.is_available(),
        }
        for cls in ALL_WRAPPERS
    ]


def _default_optimizer_values() -> List[str]:
    return [cls.name for cls in ALL_WRAPPERS if cls.is_available()]


# ---------------------------------------------------------------------------
# Public setup function
# ---------------------------------------------------------------------------

def setup_benchmark(app, app_tabs) -> None:
    """Add the Benchmark comparison tab to *app_tabs*."""

    # ---- Controls (left column) ----
    controls_card = dbc.Card([
        dbc.CardHeader(html.H5("Benchmark Settings", className="mb-0")),
        dbc.CardBody([
            dbc.Label("Test Function", style={"font-weight": "bold"}),
            dcc.Dropdown(
                id="bm-dropdown-function",
                options=_benchmark_dropdown_options(),
                value=list(BENCHMARK_REGISTRY.keys())[0],
                clearable=False,
            ),
            html.Br(),

            dbc.Label("Optimizers to compare", style={"font-weight": "bold"}),
            dbc.Checklist(
                id="bm-checklist-optimizers",
                options=_optimizer_checklist_options(),
                value=_default_optimizer_values(),
                switch=True,
            ),
            html.Br(),

            dbc.Label("Initial points (n_init)", style={"font-weight": "bold"}),
            dcc.Slider(
                id="bm-slider-n-init",
                min=3, max=20, step=1, value=5,
                marks={i: str(i) for i in range(3, 21, 3)},
                tooltip={"placement": "bottom"},
            ),
            html.Br(),

            dbc.Label("BO iterations", style={"font-weight": "bold"}),
            dcc.Slider(
                id="bm-slider-n-iter",
                min=5, max=50, step=5, value=20,
                marks={i: str(i) for i in range(5, 55, 10)},
                tooltip={"placement": "bottom"},
            ),
            html.Br(),

            dbc.Label("Random seed", style={"font-weight": "bold"}),
            dbc.Input(
                id="bm-input-seed",
                type="number", value=0,
                debounce=True,
                style={"max-width": "120px"},
            ),
            html.Br(),

            center(
                dbc.Button(
                    ["Run Benchmark ", html.I(className="bi bi-play-fill")],
                    id="bm-button-run",
                    color="primary",
                    size="lg",
                    n_clicks=0,
                )
            ),
        ]),
    ])

    # ---- Results (right column) ----
    results_card = dbc.Card([
        dbc.CardHeader(html.H5("Results", className="mb-0")),
        dbc.CardBody([
            dbc.Spinner(
                html.Div(id="bm-div-status"),
                color="primary",
                type="border",
            ),
            html.Br(),
            dcc.Graph(
                id="bm-graph-convergence",
                figure=go.Figure(
                    layout=go.Layout(
                        title="Run benchmark to see convergence",
                        xaxis_title="Evaluations",
                        yaxis_title="Best value found",
                        height=420,
                    )
                ),
            ),
            html.Br(),
            html.Div(id="bm-div-table"),
        ]),
    ])

    store = dcc.Store(id="bm-store-results", data=None)

    layout = [
        html.Br(),
        dbc.Row([
            dbc.Col(controls_card, width=4),
            dbc.Col(results_card, width=8),
        ]),
        store,
    ]

    add_tab(app_tabs, layout, "tab-benchmark", "Benchmark")
    _register_callbacks(app)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _register_callbacks(app) -> None:

    @app.callback(
        Output("bm-store-results", "data"),
        Output("bm-div-status", "children"),
        Input("bm-button-run", "n_clicks"),
        State("bm-dropdown-function", "value"),
        State("bm-checklist-optimizers", "value"),
        State("bm-slider-n-init", "value"),
        State("bm-slider-n-iter", "value"),
        State("bm-input-seed", "value"),
        prevent_initial_call=True,
    )
    def _run_benchmark(
        _n_clicks: int,
        fn_name: str,
        opt_names: List[str],
        n_init: int,
        n_iter: int,
        seed: Optional[int],
    ):
        if not opt_names:
            return None, dbc.Alert(
                "Select at least one optimizer to compare.", color="warning"
            )

        from obsidian.unified.benchmarks import get_benchmark
        from obsidian.unified.benchmark_runner import run_comparison

        benchmark = get_benchmark(fn_name)
        wrapper_map = {cls.name: cls for cls in ALL_WRAPPERS}
        optimizers = [
            wrapper_map[name]()
            for name in opt_names
            if name in wrapper_map and wrapper_map[name].is_available()
        ]
        if not optimizers:
            return None, dbc.Alert(
                "None of the selected optimizers are available in this environment.",
                color="danger",
            )

        try:
            results = run_comparison(
                optimizers=optimizers,
                benchmark=benchmark,
                n_init=n_init,
                n_iterations=n_iter,
                n_suggest=1,
                seed=int(seed) if seed is not None else 0,
            )
        except Exception as exc:  # noqa: BLE001
            return None, dbc.Alert(f"Error during benchmark: {exc}", color="danger")

        # Serialise (drop DataFrames which aren’t JSON-safe)
        store_data = [
            {
                "optimizer": r["optimizer"],
                "best_per_eval": r["best_per_eval"],
                "n_evals": r["n_evals"],
                "best_found": float(
                    max(r["best_per_eval"]) if not benchmark.minimize
                    else min(r["best_per_eval"])
                ),
                "total_evals": len(r["n_evals"]),
            }
            for r in results
        ]

        n_opt = len(store_data)
        status = dbc.Alert(
            f"Completed {n_opt} optimizer(s) on '{fn_name}' "
            f"({n_init} init + {n_iter} BO iterations each).",
            color="success",
        )
        return store_data, status

    @app.callback(
        Output("bm-graph-convergence", "figure"),
        Output("bm-div-table", "children"),
        Input("bm-store-results", "data"),
        State("bm-dropdown-function", "value"),
        prevent_initial_call=True,
    )
    def _update_plots(store_data: Optional[List[Dict]], fn_name: str):
        if not store_data:
            return go.Figure(), html.Div()

        from obsidian.unified.benchmarks import get_benchmark
        import pandas as pd

        benchmark = get_benchmark(fn_name)

        # -- convergence plot --
        fig = go.Figure()
        for r in store_data:
            fig.add_trace(go.Scatter(
                x=r["n_evals"],
                y=r["best_per_eval"],
                mode="lines+markers",
                name=r["optimizer"],
                line={"width": 2},
                marker={"size": 5},
            ))

        if benchmark.optimum is not None:
            fig.add_hline(
                y=benchmark.optimum,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Optimum ({benchmark.optimum:.4g})",
                annotation_position="top right",
            )

        direction = "(min)" if benchmark.minimize else "(max)"
        fig.update_layout(
            title=f"Convergence — {fn_name}",
            xaxis_title="Number of evaluations",
            yaxis_title=f"Best value found {direction}",
            legend_title="Optimizer",
            height=450,
            margin={"t": 50},
        )

        # -- summary table --
        summary = pd.DataFrame([
            {
                "Optimizer": r["optimizer"],
                "Best Found": round(r["best_found"], 6),
                "Total Evaluations": r["total_evals"],
                "Gap to Optimum": (
                    round(abs(benchmark.optimum - r["best_found"]), 6)
                    if benchmark.optimum is not None else "N/A"
                ),
            }
            for r in store_data
        ])
        table = center(make_table(summary, fill_width=True))

        return fig, table
