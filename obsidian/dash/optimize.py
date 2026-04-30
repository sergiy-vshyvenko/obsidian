from .utils import add_tab, make_input, make_dropdown, make_switch, make_slider, make_knob, make_table, make_collapse
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table, callback, Output, Input, State, ALL, MATCH

import pandas as pd

from obsidian.parameters import ParamSpace, Param_Categorical, Param_Ordinal, Param_Continuous
from obsidian.optimizer import BayesianOptimizer
from obsidian.campaign import Campaign
from obsidian.parameters import Target

from obsidian.plotting.plotly import parity_plot
from .utils import load_optimizer, center

from obsidian.unified.optimizer_wrappers import get_available_optimizers


# Server-side wrapper cache (single-user local app)
_active_wrapper = {"instance": None}


def _get_responses(config):
    """Return [{name, aim}] from config."""
    responses = (config or {}).get('responses')
    if responses:
        return responses
    name = (config or {}).get('response_name')
    return [{'name': name, 'aim': 'max'}] if name else []


def setup_optimize(app, app_tabs):

    available = get_available_optimizers()
    backend_options = [{"label": w.label, "value": w.name} for w in available]
    default_backend = backend_options[0]["value"] if backend_options else "obsidian"

    backend_selector = dbc.Container(
        dbc.Row(
            dbc.Col(
                make_dropdown(
                    "Optimizer Backend",
                    "Select the optimization library to use for fitting and candidate suggestions",
                    backend_options,
                    id="input-optimizer_backend",
                    kwargs={"value": default_backend},
                ),
                width=4,
            )
        ),
        style={"margin-bottom": "10px"},
    )

    fit_div = dbc.Container([
        dbc.Spinner(color="primary", children=[
            dbc.Button('Fit', id='button-fit', n_clicks=0, size='lg')
        ]),
        html.Br(),
        html.Br(),
        dbc.Card([
            dbc.CardHeader('Regression Statistics'),
            dbc.CardBody([
                html.Div(id='div-fit', children=[])
                ]),
            ]),
        html.Div(id='graph-parity', children=[])
        ],
        style={'textAlign': 'center'}
    )

    predict_div = dbc.Container([
        dbc.Spinner(color="primary", children=[
            dbc.Button('Optimize', id='button-predict', n_clicks=0, size='lg'),
            ]),
        html.Br(),
        html.Br(),
        dbc.Card([
            dbc.CardHeader('Optimal Experiments'),
            dbc.CardBody([
                html.Div(id='div-predict', children=[], style={})
                ]),
            ]),
        ],
        style={'textAlign': 'center'}
    )

    storage_fit = dcc.Store(id='store-fit', data=None)

    store_candidates = dcc.Store(id='store-candidates', data={})

    candidates_downloader = html.Div(children=[
        dbc.Button('Download Suggested Candidates', id='button-download_candidates',
                   className='me-2', color='primary'),
        dcc.Download(id='downloader-candidates')],
                                     style={'textAlign': 'center', 'margin-top': '15px'})

    columns = dbc.Row([dbc.Col(fit_div, width=6), dbc.Col([predict_div, candidates_downloader], width=6)])
    elements = [html.Br(), backend_selector, columns, storage_fit, store_candidates]
    add_tab(app_tabs, elements, 'tab-optimize', 'Optimize')
    setup_optimize_callbacks(app)

    return


def setup_optimize_callbacks(app):

    @app.callback(
        Output('button-fit', 'n_clicks'),
        Output('store-fit', 'data'),
        Input('button-fit', 'n_clicks'),
        State('store-config', 'data'),
        State('store-X0', 'data'),
        State('store-Xspace', 'data'),
        State('input-optimizer_backend', 'value'),
        prevent_initial_call=True
    )
    def fit_optimizer(fit_clicked, config, X0, Xspace_save, backend):

        if not Xspace_save:
            return 0, None

        backend = backend or "obsidian"
        responses = _get_responses(config)
        if not responses:
            return 0, None

        X_space = ParamSpace.load_state(Xspace_save)
        df = pd.DataFrame(X0)

        if backend == "obsidian":
            targets = [Target(r['name'], aim=r['aim']) for r in responses]
            campaign = Campaign(X_space, targets if len(targets) > 1 else targets[0])
            campaign.add_data(df)
            optimizer = BayesianOptimizer(X_space=campaign.X_space,
                                          surrogate=config['surrogate_params']['surrogate'])
            campaign.set_optimizer(optimizer)
            campaign.fit()
            return 0, {"backend": "obsidian", "state": campaign.optimizer.save_state()}

        # BoFire / BayBe — extract continuous bounds and categorical params
        param_bounds = {p.name: (p.min, p.max) for p in X_space.params
                        if isinstance(p, Param_Continuous)}
        param_categories = {p.name: list(p.categories) for p in X_space.params
                            if isinstance(p, (Param_Categorical, Param_Ordinal))}

        wrapper_map = {w.name: w for w in get_available_optimizers()}
        WrapperClass = wrapper_map.get(backend)
        if WrapperClass is None:
            return 0, None

        objectives = [(r['name'], r['aim'] == 'min') for r in responses]
        wrapper = WrapperClass()
        wrapper.setup(param_bounds, param_categories=param_categories or None,
                      objectives=objectives)

        all_param_names = list(param_bounds.keys()) + list(param_categories.keys())
        X = df[all_param_names]
        response_names = [r['name'] for r in responses]
        y = df[response_names] if len(response_names) > 1 else df[response_names[0]]
        wrapper.fit(X, y)

        _active_wrapper["instance"] = wrapper
        return 0, {"backend": backend, "fitted": True,
                   "param_names": all_param_names,
                   "response_names": response_names}

    @app.callback(
        Output('div-fit', 'children'),
        Input('store-fit', 'data'),
        State('store-config', 'data'),
        State('uploader-X0', 'filename')
    )
    def fit_statistics(opt_save, config, filename):
        if opt_save is None:
            return dbc.Alert('Model must be fit first', color='info')

        backend = opt_save.get("backend", "obsidian") if isinstance(opt_save, dict) else "obsidian"

        if backend == "obsidian":
            state = opt_save["state"] if isinstance(opt_save, dict) else opt_save
            optimizer = load_optimizer(config, state)
            responses = _get_responses(config)
            items = [
                dbc.ListGroupItem(['Model Type: ', f'{optimizer.surrogate_type}']),
                dbc.ListGroupItem(['Data Name: ', filename]),
            ]
            for i, surr in enumerate(optimizer.surrogate):
                label = responses[i]['name'] if i < len(responses) else f'Response {i+1}'
                items += [
                    dbc.ListGroupItem([f'{label} R', html.Sup('2'), ': ', f'{surr.r2_score:.4g}']),
                    dbc.ListGroupItem([f'{label} MLL: ', f'{surr.loss:.4g}']),
                ]
            return dbc.ListGroup(items, flush=True)

        backend_labels = {"bofire": "BoFire (BoTorch)", "baybe": "BayBe (BoTorch)"}
        response_names = opt_save.get("response_names", [opt_save.get("response_name", "")])
        return dbc.ListGroup([
            dbc.ListGroupItem(['Backend: ', backend_labels.get(backend, backend)]),
            dbc.ListGroupItem(['Data Name: ', filename]),
            dbc.ListGroupItem(['Responses: ', ', '.join(response_names)]),
            dbc.ListGroupItem('Surrogate fitted successfully'),
        ], flush=True)

    @app.callback(
        Output('graph-parity', 'children'),
        Input('store-fit', 'data'),
        State('store-config', 'data'),
        prevent_initial_call=True
    )
    def graph_parity_plot(opt_save, config):
        if opt_save is None:
            return None

        backend = opt_save.get("backend", "obsidian") if isinstance(opt_save, dict) else "obsidian"
        if backend != "obsidian":
            return None

        state = opt_save["state"] if isinstance(opt_save, dict) else opt_save
        optimizer = load_optimizer(config, state)
        pplot = parity_plot(optimizer)
        pplot.update_layout(height=400, width=600)
        return center(dcc.Graph(figure=pplot))

    @app.callback(
        Output('div-predict', 'children'),
        Output('div-predict', 'style'),
        Output('button-predict', 'n_clicks'),
        Output('store-candidates', 'data'),
        Input('button-predict', 'n_clicks'),
        State('store-config', 'data'),
        State('store-fit', 'data'),
        State('input-optimizer_backend', 'value'),
    )
    def predict_optimizer(predict_clicked, config, opt_save, backend):

        backend = backend or "obsidian"

        if opt_save is None:
            alert_color = 'danger' if predict_clicked else 'info'
            return dbc.Alert('Model must be fit first', color=alert_color), {}, predict_clicked, {}

        fitted_backend = opt_save.get("backend", "obsidian") if isinstance(opt_save, dict) else "obsidian"
        if fitted_backend != backend:
            return (
                dbc.Alert('Backend changed since last fit — please re-fit the model', color='warning'),
                {}, predict_clicked, {}
            )

        if backend == "obsidian":
            responses = _get_responses(config)
            state = opt_save["state"] if isinstance(opt_save, dict) else opt_save
            optimizer = load_optimizer(config, state)
            aq_params = dict(config['aq_params'])
            if len(responses) > 1:
                aq_params['acquisition'] = ['NEHVI']
            X_suggest, eval_suggest = optimizer.suggest(**aq_params)
            df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
            df_suggest.insert(loc=0, column='CandidatesID', value=df_suggest.index)
            tables = [center(make_table(df_suggest))]
            return tables, {'overflow-x': 'scroll'}, 0, df_suggest.to_dict()

        wrapper = _active_wrapper.get("instance")
        if wrapper is None:
            return dbc.Alert('No fitted model found — please re-fit', color='danger'), {}, 0, {}

        m_batch = config['aq_params'].get('m_batch', 1) if config else 1
        X_suggest = wrapper.suggest(n=m_batch)
        if X_suggest is None:
            return dbc.Alert('Optimizer failed to produce suggestions', color='danger'), {}, 0, {}

        X_suggest.insert(loc=0, column='CandidatesID', value=X_suggest.index)
        tables = [center(make_table(X_suggest))]
        return tables, {'overflow-x': 'scroll'}, 0, X_suggest.to_dict()

    @app.callback(
        Output('downloader-candidates', 'data'),
        Input('button-download_candidates', 'n_clicks'),
        State('store-candidates', 'data'),
        prevent_initial_call=True
    )
    def download_candidates(n_clicks, data):
        df = pd.DataFrame(data)
        return dcc.send_data_frame(df.to_csv, 'Suggested_Candidates.csv', index=False)

    return
