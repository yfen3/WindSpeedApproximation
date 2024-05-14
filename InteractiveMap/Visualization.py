# Get the data for expeirment
from InteractiveMap.Info import get_info_panel
from InteractiveMap.Map import get_map
from InteractiveMap.Tooltip import get_tooltip
import InteractiveMap.Utils as Utils
from InteractiveMap.DataLoader import DataLoader
from InteractiveMap.ModelLoader import ModelLoader
from sklearn.preprocessing import StandardScaler
from dash import Dash, html, Input, Output, callback, exceptions
import dash_leaflet as dl
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
    'background': 'grey',
    'text': 'yellow'
}

# How many stations are used to predict
number_of_stations = 9#11

# Build the basic UI elements
header = html.H4(children='Visualization of the FES project', style={'font_color': colors['text']})
info = get_info_panel('Click on map to view the wind speed prediction', {'font_color': colors['text']})
tooltip = get_tooltip([170, 180], 'Wind Speed: 10 km/h')
map = get_map(children=[tooltip])

# TODO Refactor this into a separate file
df = pd.read_csv('Data/processed_ab_wind_test.txt')
latest_one_month_records = df.sort_values('date', ascending=False).head(20*300)
data_loader = DataLoader()
data_loader.load_data_from_dataframe(latest_one_month_records)

def get_prediction_model(data_loader):
    # Load the model
    nn_model = keras.models.load_model('Visualization_proj_9stations.keras')

    nn_model_loader = ModelLoader()
    nn_model_loader.load_model(nn_model)

    return nn_model_loader

nn_model_loader = get_prediction_model(data_loader)


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

    #The DOM update is not triggered if the id stays the same
    #however, re-iniitalize the tooltip is necessary to change the content
    #so an id with random number is used to force the update
    random_number = np.random.randint(0, 100)
    
    tooltip_content = dl.Tooltip(
                id=f'tooltip_content{random_number}',
                content=f'Wind Speed: {wind_speed} km/h', 
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
        return f'Output: {lat}, {lng}'


# Make the app and run it
# the basic layout of the app is also defined here
app = Dash(__name__)
app.layout = html.Div(style={'backgroundColor': colors['background']}, 
                      children=[
                          header,
                          info,
                          map,
])
app.run_server(debug=True)
