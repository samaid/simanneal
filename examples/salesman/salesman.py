import dash
from impl_tsp_vanilla import *
import pandas as pd
from dash import dcc, html, ctx, dash_table
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import threading

from utils import *
from simanneal import TravellingSalesmanProblem
import time

global_best_energy = global_best_time = global_trials_per_second = 0.0
global_tsp = None
global_impl = None
global_play = False
global_pause_start = 0.0
global_time_spent_in_pause = 0.0
global_time_spent_in_compute = 0.0
global_status = "Auto-Tuning..." # "Running...", "Paused", "Auto-Tuning...", "Completed"


def auto_tune_tsp():
    global global_best_energy, global_best_time, global_trials_per_second, global_tsp

    # Prepare data for TSP Annealer
    coordinates, dm = create_distance_matrix(df)
    matprint(dm)
    cities = df.City.values
    print(cities)
    init_state = np.arange(N_CITIES)


    # Warm-up annealer
    if global_impl == IMPLEMENTATIONS[0]:
        global_tsp = TravellingSalesmanProblemVanilla(init_state, dm)
    else:
        global_tsp = TravellingSalesmanProblem(init_state, dm)
    global_tsp.set_schedule(global_tsp.auto(minutes=2.0))
    global_tsp.state = global_tsp.copy_state(init_state)  # Reset state after initial warm-up
    global_tsp.best_state = global_tsp.copy_state(init_state ) # Reset state after initial warm-up
    global_best_energy = global_tsp.energy()
    global_best_time = time.time() - global_tsp.start
    global_trials_per_second = 0.0


def htmlHeader():
    return dbc.Row(dbc.Col([
        html.H4(f"Finding optimal traveling route across top {N_CITIES} US cities", style={'text-align': 'center'}),
        html.H6("Traveling Salesman Problem solution using Simulated Annealing", style={'text-align': 'center'}),
    ]))


def htmlSelectors():
    return dbc.Row(
        [
            dbc.Col(html.Div(
                [
                    html.Label("Select implementation variant:"),
                    html.Div(dcc.Dropdown(IMPLEMENTATIONS, IMPLEMENTATIONS[0], id='dd-impl'),
                             style={"width": "200px"}),
                ], style={"padding": "10px"},
            ), width=2),
            dbc.Col(html.Div(
                [
                    html.Br(),
                    html.Button("Play/Pause", id='btn-play-pause', n_clicks=0),
                    html.Label(id='output-status', style={"color": "#F0A010", "padding": "5px"}),
                ], style={"padding": "10px"}
            ), width=9),
        ]
    )


def htmlGraphTable():
    df = pd.DataFrame(columns=["Trial path (mi)", "Best path (mi)",
                               "Found in (sec)", "Trials/ sec", "Total Elapsed (sec)"])
    df.loc[0] = [0, 0, 0, 0, 0]

    return dbc.Row([
        dbc.Col(dcc.Graph(id='map-graph', config={'displayModeBar': False}), width=9),
        dbc.Col(dash_table.DataTable(df.to_dict("records"), [{"name": c, "id": c} for c in df.columns],
                                     id="output-table",
                                     style_as_list_view=True,
                                     style_cell={"padding": "5px", "textAlign": "center", "width": "10%"},
                                     style_header={
                                         'backgroundColor': "lightgrey",
                                         'fontWeight': "semi-bold",
                                         'whiteSpace': 'normal',
                                         'height': 'auto'
                                     },
                                     ), width=2,),
        dbc.Col(html.Div(id="dd-output-container"), width=1)
    ])


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    [
        htmlHeader(),
        htmlSelectors(),
        html.Br(),
        htmlGraphTable(),

        dcc.Interval(
            id='graph-update',
            interval=400,
            n_intervals=0
        )
    ], style={"font-size": "80%"}
)


@app.callback(
    Output('output-status', 'children'),
    Output('dd-impl', 'disabled'),
    Input('btn-play-pause', 'n_clicks'),
)
def btnClick(btn):
    global global_play, global_status

    if ctx.triggered_id == 'btn-play-pause':
        global_play = ~global_play
        if global_play:
            global_status = "Playing..."
        else:
            global_status = "Paused"

    s = f"STATUS: {global_status}"
    return [s, global_play]


@app.callback(
    Output('dd-output-container', 'children'),
    Input('dd-impl', 'value')
)
def update_implementation(value):
    global global_impl, global_tsp
    if global_impl != value:
        global_impl = value
        global_tsp.user_exit = True
    return ""  # Return nothing


@app.callback(
        Output('output-table', 'data'),
        Output('map-graph', 'figure'),
#        Output('output-status', 'children'),
#        Output('dd-impl', 'disabled'),
        Output('btn-play-pause', 'disabled'),
        Input('graph-update', 'n_intervals')
)
def update(n):
    global global_tsp, global_play, global_status

    glabal_play = global_status == "Playing..."

    if global_tsp is None:
        out_energy = "--"
        out_best_energy = "--"
    else:
        if hasattr(global_tsp, "E"):
            out_energy = "--"
        else:
            out_energy = f"{human_format(global_tsp.E)}"
        if hasattr(global_tsp, "best_energy"):
            out_best_energy = "--"
        else:
            out_best_energy = f"{human_format(global_tsp.best_energy)}"
    out_best_time = f"{global_best_time:.1f}"
    out_throughput = f"{human_format(global_trials_per_second)}"
    out_elapsed = f"{global_time_spent_in_compute:.1f}"

    df = pd.DataFrame(columns=["Trial path (mi)", "Best path (mi)",
                               "Found in (sec)", "Trials/ sec", "Total Elapsed (sec)"])
    df.loc[0] = [out_energy, out_best_energy, out_best_time, out_throughput, out_elapsed]

    fig = create_figure(global_tsp)
    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(x=0.6)
    )

    out_status = f"STATUS: {global_status}"
    out_disable_impl = global_status in ["Auto-Tuning...", "Playing..."]
    out_disable_btn = global_status in ["Auto-Tuning..."]

    return [df.to_dict('records'), fig, out_disable_btn]
#    return [df.to_dict('records'), fig, out_status, out_disable_impl, out_disable_btn]


def tsp_updater(tsp, step, trials, accepts, improves, dt):
    global global_best_energy, global_best_time, global_trials_per_second
    global global_play
    global global_pause_start, global_time_spent_in_pause, global_time_spent_in_compute

    if tsp.pause == global_play:
        # Here if Play/Pause button wos pressed
        if global_play:
            # Resume playing
            global_time_spent_in_pause += time.time() - global_pause_start
        else:
            # Pause
            global_pause_start = time.time()

    tsp.pause = ~global_play

    if global_play:
        global_time_spent_in_compute = time.time() - tsp.start - global_time_spent_in_pause
        if tsp.best_energy < global_best_energy:
            global_best_energy = tsp.best_energy
            global_best_time = time.time() - tsp.start
        if dt > 0:
            global_trials_per_second = trials/dt
        else:
            global_trials_per_second = 0.0

    time.sleep(0.01)  # Allow GIL to switch to another thread to properly visualize data


def visualize():
    app.run_server()


def compute():
    global global_tsp, global_play, global_status

    while not global_play:
        time.sleep(0.01)

    global_tsp.anneal(tsp_updater)
    global_play = False
    global_status = "Completed"


def restart():
    global global_play, global_status
    global_play = False
    global_status = "Auto-Tuning..."
    auto_tune_tsp()
    global_status = "Paused"
    vis_task = threading.Thread(target=visualize)
    vis_task.start()
    comp_task = threading.Thread(target=compute)
    comp_task.start()


def main():
    global global_tsp
    while True:  # Infinite loop of restarts for different implementation settings
        restart()
        while not global_tsp.user_exit:
            time.sleep(0.01)


if __name__ == '__main__':
    main()
