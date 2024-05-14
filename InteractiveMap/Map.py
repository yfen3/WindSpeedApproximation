import pandas as pd 
import dash_leaflet as dl
from dash import Dash, dcc, html, Input, Output, callback, exceptions

def get_wind_data():
    plot_data = pd.read_csv('Data/processed_ab_wind_test.txt')
    map_data = plot_data.groupby('name').head(1)

    return map_data

def get_map(children):
    base_layer = dl.TileLayer(id="TileMap")

    map = dl.Map(
        children=[
            base_layer,
            *children,
            ],
        style={'width': '500px', 'height': '500px'},
        center=[53.52, -113.40],
        maxBounds=[[49,-119], [55,-109]],
        zoom=5,
        minZoom=5,
        maxZoom=10,
        id='map',
               )

    return map