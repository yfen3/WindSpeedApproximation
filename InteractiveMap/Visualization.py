# Get the data for expeirment
from dash import Dash, html, Input, Output, callback, exceptions, dcc
import dash_leaflet as dl
import Utils as Utils
from Info import get_info_panel
from Map import get_map
from Tooltip import get_tooltip
from Graph import get_graph
from AbWindDataModel import AbWindDataModel
import pandas as pd
# TODO Refactor this into a separate file
import torch
import matplotlib.pyplot as plt
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import numpy as np

device = torch.device("cuda:0")
print(f"Keras version is {keras.__version__}")
print(f"Num GPUs Available: {torch.cuda.device_count()}")
# End of TODO

# Define the style of the app
colors = {
    'background': '#74fcea',
    'text': '#AF0038'
}

# How many stations are used to predict
number_of_stations = 9

# Build the basic UI elements
header = html.H2(children='Visualization of the FES project', style={'color': colors['text']})
info = get_info_panel(
    'Click on map to view the wind speed prediction', 
    {
        'color': colors['text'],
        'fontSize': '20px',
        }
    )
tooltip = get_tooltip([170, 180], 'Wind Speed: 10 km/h')
map = get_map(children=[tooltip])
# use browser session to store the data
store = dcc.Store(id='store', storage_type='session')

data_loader = AbWindDataModel().get_latest_one_month_records_data_loader()
nn_model_loader = AbWindDataModel().get_prediction_model()

df = data_loader.get_data()
graph = html.Div(children=[], id='monthly-wind-graph')

# TODO see if those callbacks can be moved to a separate file
# Define callbacks
@callback(
        Output(component_id="tooltip", component_property="position"), 
        Input(component_id="map", component_property="clickData"), 
        prevent_initial_call=True
        )
def show_tool_tip_at_click_position(onClickData):
    lat, lng = Utils.get_latlng(onClickData)
    position = [lat, lng]

    return position

@callback(
        Output(component_id="tooltip", component_property="children"), 
        Input(component_id="map", component_property="clickData"), 
        prevent_initial_call=True
        )
def show_tooltip_content(onClickData):
    lat, lng = Utils.get_latlng(onClickData)
    wind_speed = Utils.get_wind_speed(lat, lng, data_loader, nn_model_loader, number_of_stations)
    most_recent_wind_speed = Utils.get_most_recent_model_prediction(wind_speed)

    #The DOM update is not triggered if the id stays the same
    #however, re-iniitalize the tooltip is necessary to change the content
    #so an id with random number is used to force the update
    random_number = np.random.randint(0, 100)
    
    tooltip_content = dl.Tooltip(
                id=f'tooltip_content{random_number}',
                content=f'Wind Speed: {str(most_recent_wind_speed)} km/h', 
                permanent=True,
                )
    
    return tooltip_content

@callback(
        Output(component_id="info", component_property="children"), 
        Input(component_id='map', component_property='clickData'),
        )
def show_click_info(onClickData):
    if onClickData is None:
        raise exceptions.PreventUpdate
    else:
        lat, lng = Utils.get_latlng(onClickData)
        return f'Location clicked: {np.round(lat, 4)}, {np.round(lng, 4)}'
    
@callback(
        Output(component_id="monthly-wind-graph", component_property="children"), 
        Input(component_id='map', component_property='clickData'),
        )
def show_monthly_graph(onClickData):
    if onClickData is None:
        raise exceptions.PreventUpdate
    else:
        lat, lng = Utils.get_latlng(onClickData)
        wind_speed = Utils.get_wind_speed(lat, lng, data_loader, nn_model_loader, number_of_stations)
        dates = df['date'].unique()

        new_df = pd.DataFrame(
            dict(
                date = dates, 
                wind_speed = wind_speed
            )
            ) 
        graph = get_graph('Monthly Wind Speed Prediction', new_df, 'date', 'wind_speed')
        return graph

# UI elements
graph_section = html.Div(
    children=[map, graph], 
    style={
        'display': 'flex', 
        'flexDirection':'row', 
        'gap' : '10px',
        'padding-top': '10px',
        'backgroundColor': colors['background']
        }
    )

# Make the app and run it
# the basic layout of the app is also defined here
app = Dash(__name__)
app.layout = html.Div(
    style={
    'backgroundColor': colors['background'],
    'padding': '10px'
    }, 
    children=[
        header,
        info,
        graph_section
])
app.run_server(debug=True)
