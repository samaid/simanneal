import plotly.graph_objects as go
import pandas as pd

# 'https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv'
df_airports = pd.read_csv(
    'https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv'
)
df_airports.set_index('cnt', inplace=True)
df_airports.sort_index(ascending=False, inplace=True)
df_airports = df_airports.head(20).reset_index()
print(df_airports)

df_flight_paths = pd.read_csv(
    'https://raw.githubusercontent.com/plotly/datasets/master/2011_february_aa_flight_paths.csv')
df_flight_paths.set_index('cnt', inplace=True)
df_flight_paths.sort_index(ascending=False, inplace=True)
df_flight_paths = df_flight_paths.head(20).reset_index()
print(df_flight_paths)

fig = go.Figure(go.Scattergeo())

fig.add_trace(go.Scattergeo(
    showlegend=False,
    locationmode = 'USA-states',
    lon = df_airports['long'],
    lat = df_airports['lat'],
    hoverinfo = 'text',
    text = df_airports['airport'],
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgb(255, 0, 0)',
        line = dict(
            width = 3,
            color = 'rgba(68, 68, 68, 0)'
        )
    )))

fig.update_geos(
    visible=True, resolution=50, scope="usa",
    showcountries=True, countrycolor="rgb(204, 204, 204)", landcolor="rgb(255, 255, 240)",
    showsubunits=True, subunitcolor="rgb(160, 160, 160)"
)
fig.update_layout(height=1200, margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
