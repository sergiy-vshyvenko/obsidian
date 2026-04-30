"""Unified multi-optimizer APO server.

Run::

    python app_unified.py

The server starts on http://127.0.0.1:8050 by default.
"""

import pandas as pd
from dash import Dash, html
import dash_bootstrap_components as dbc
from PIL import Image

from obsidian.dash import (
    setup_data,
    setup_config,
    setup_optimize,
    setup_plots,
    setup_predict,
    setup_infobar,
)

from obsidian.parameters import (
    ParamSpace,
    Param_Categorical,
    Param_Ordinal,
    Param_Continuous,
)
from obsidian.experiment import ExpDesigner, Simulator
from obsidian.experiment.benchmark import shifted_parab


# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)

dbc_css = (
    "https://cdn.jsdelivr.net/gh/AnnMarieW/"
    "dash-bootstrap-templates/dbc.min.css"
)
app.config.external_stylesheets = [
    dbc.themes.SANDSTONE,
    dbc_css,
    dbc.icons.BOOTSTRAP,
]

logo = Image.open("docs/_static/obsidian_logo.png")

app_image = html.Div(
    html.Img(src=logo, style={"width": "5%", "height": "5%"}),
    style={"textAlign": "center"},
)
app_title = html.Div(
    [
        html.H1("obsidian"),
        html.H5(
            "Algorithmic Process Optimisation — Multi-Optimizer Benchmark",
            style={"font-style": "italic", "color": "#AAAAAA"},
        ),
    ],
    style={"textAlign": "center"},
)
app_infobar = html.Div(id="root-infobar")
app_tabs = dbc.Tabs(children=[], id="root-tabs")

app.title = "obsidian Unified APO Server"
app.layout = dbc.Container(
    children=[app_image, app_title, app_infobar, app_tabs],
    id="root-div",
    fluid=True,
)


# ---------------------------------------------------------------------------
# Default example data (same as the standard app.py)
# ---------------------------------------------------------------------------

params = [
    Param_Continuous("Temperature", -10, 30),
    Param_Continuous("Concentration", 10, 150),
    Param_Continuous("Enzyme", 0.01, 0.30),
    Param_Categorical("Variant", ["MRK001", "MRK002", "MRK003"]),
    Param_Ordinal("Stir Rate", ["0 - Low", "1 - Medium", "2 - High"]),
]
X_space = ParamSpace(params)
designer = ExpDesigner(X_space, seed=0)
X0 = designer.initialize(10, "LHS")
simulator = Simulator(X_space, shifted_parab, name="Yield", eps=0.05)
y0 = simulator.simulate(X0)
default_data = pd.concat([X0, y0], axis=1)


# ---------------------------------------------------------------------------
# Register tabs
# ---------------------------------------------------------------------------

setup_infobar(app, app_infobar)
setup_data(app, app_tabs, default_data, X_space)
setup_config(app, app_tabs)
setup_optimize(app, app_tabs)
setup_plots(app, app_tabs)
setup_predict(app, app_tabs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
