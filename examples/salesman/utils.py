import math
import numpy as np
import plotly.graph_objects as go
import pandas as pd


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def distance(a, b):
    """Calculates distance between two latitude-longitude coordinates."""
    R = 3963.0  # radius of Earth (miles)
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    return math.acos(math.sin(lat1) * math.sin(lat2) +
                     math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2)) * R


def create_distance_matrix(df):
    lat = df.lat.values
    lon = df.lon.values
    n = df.index.size
    coordinates = list(zip(lat, lon))
    distance_matrix = np.empty(shape=(n,n))

    for i, a in enumerate(coordinates):
        for j, b in enumerate(coordinates):
            distance_matrix[i, j] = 0.0 if i == j else distance(a, b)

    return coordinates, distance_matrix


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

# CONSTANTS
N_CITIES = 52  # Total number of cities
ANNEAL_TIME = 1.0  # Total annealing time in minutes
IMPLEMENTATIONS = ["vanilla", "numpy", "mkl_random", "parallel mkl_random", "dpnp"]

# Read dataset
dataset_file = 'https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv'
df = pd.read_csv(dataset_file)
df.set_index('Population', inplace=True)
df.sort_index(ascending=False, inplace=True)
df = df.head(N_CITIES).reset_index()
print(df)


def add_trial_path(fig, tsp):
    lat = df.lat.values
    lon = df.lon.values

    lat_trace = list()
    lon_trace = list()
    for i in range(-1, len(tsp.state)):
        lat_trace.append(lat[tsp.state[i]])
        lon_trace.append(lon[tsp.state[i]])

    fig.add_trace(
        go.Scattergeo(
            name="Trial route",
            showlegend=True,
            locationmode='USA-states',
            lon=lon_trace,
            lat=lat_trace,
            mode='lines',
            line=dict(width=1,color="rgb(128, 128, 128)"),
        )
    )


def add_best_path(fig, tsp):
    lat = df.lat.values
    lon = df.lon.values

    lat_trace = list()
    lon_trace = list()
    for i in range(-1, len(tsp.state)):
        lat_trace.append(lat[tsp.best_state[i]])
        lon_trace.append(lon[tsp.best_state[i]])

    fig.add_trace(
        go.Scattergeo(
            name="Best route",
            showlegend=True,
            locationmode='USA-states',
            lon=lon_trace,
            lat=lat_trace,
            mode='lines',
            line=dict(width=2,color="rgb(255, 128, 128)"),
        )
    )


def create_figure(tsp):
    fig = go.Figure(go.Scattergeo())

    fig.update_geos(
        visible=True, resolution=50, scope="usa",
        showcountries=True, countrycolor="rgb(204, 204, 204)", landcolor="rgb(255, 255, 240)",
        showsubunits=True, subunitcolor="rgb(160, 160, 160)")

    add_trial_path(fig, tsp)
    add_best_path(fig, tsp)

    fig.add_trace(go.Scattergeo(
        name="City",
        showlegend=True,
        locationmode='USA-states',
        lon=df['lon'],
        lat=df['lat'],
        hoverinfo='text',
        text=df['City'],
        mode='markers',
        marker=dict(
            size=10,
            color='rgb(0, 0, 0)',
            line=dict(
                width=3,
                color='rgba(68, 68, 68, 0)'
            )
        )))

    return fig
