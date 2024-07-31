# Get the data for expeirment

from dash import Dash, html, Input, Output, State, callback, exceptions, dcc
import dash_leaflet as dl
import Utils as Utils
from Info import get_info_panel
from Map import get_map
from Tooltip import get_tooltip
from Graph import get_graph
from AbWindDataModel import AbWindDataModel
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import numpy as np
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError
import time

device = torch.device("cuda:0")
print(f"Keras version is {keras.__version__}")
print(f"Num GPUs Available: {torch.cuda.device_count()}")

# Load the data and models
data_loader = AbWindDataModel().get_latest_one_month_records_data_loader()
nn_model_loader = AbWindDataModel().get_nn_model()
ts_model_loader = AbWindDataModel().get_ts_model()
gp_martern_model_loader = AbWindDataModel().get_gp_martern_model()
gp_martern_ts_model_loader = AbWindDataModel().get_gp_martern_ts_model()

model_loaders = {
    'NN': nn_model_loader,
    'TS': ts_model_loader,
    'GP (Martern)': gp_martern_model_loader,
    'GP TS (Martern)': gp_martern_ts_model_loader
}

df = data_loader.get_data()

# Define the style of the app
colors = {
    'background': '#74fcea',
    'text': '#AF0038'
}

# How many stations are used to predict
# This number is predetermined as the model need to be pretrained for the number of stations
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

model_selector =  dcc.Dropdown(
    list(model_loaders.keys()), 
    'NN', 
    id='model-dropdown',
    style={'width': '300px'}
    )

graph_selector =  dcc.Dropdown(
    ['Lowerbound Prediction', 'Upperbound Prediction'], 
    'Upperbound Prediction', 
    id='graph-dropdown',
    style={'width': '300px'}
    )

search_bar = dcc.Input(
    id='search-bar',
    type='text',
    placeholder='Enter a location...',
    style={'width': '300px'}
)

search_button = html.Button(
    'Search',
    id='search-button',
    n_clicks=0,
    style={'margin-left': '10px'}
)

graph = html.Div(children=[], id='detail-graph')
# use browser session to store the data
store = dcc.Store(id='store', storage_type='session')

# TODO see if those callbacks can be moved to a separate file
# Define callbacks
@callback(
        Output(component_id="store", component_property="data"), 
        [Input(component_id="model-dropdown", component_property="value"), 
         Input('search-button', 'n_clicks'),],
        [State(component_id="store", component_property="data"),
         State('search-bar', 'value'),],

        prevent_initial_call=True
        )
def update_store_data(model_value, n_clicks, store_data, search_value):
    if isinstance(store_data, str):
        store_data = {}
    elif store_data is None:
        store_data = {}

    if search_value is not None and search_value != "":
        store_data['search'] = getCoordinates(search_value)

    if model_value is not None:
        store_data['model'] = model_value
    
    return store_data


@callback(
        Output(component_id="tooltip", component_property="position"), 
        [Input(component_id="map", component_property="clickData"),
         Input('search-button', 'n_clicks'),],
        State('search-bar', 'value'),
        State(component_id="store", component_property="data"),
         
        prevent_initial_call=True
        )
def show_tool_tip(mapOnClickData, n_clicks, search_value, data):
    if search_value is not None and search_value != "":
        position = getCoordinates(search_value)
        if position != "":
            return position
    
    if mapOnClickData is not None:
        lat, lng = Utils.get_latlng(mapOnClickData)
        if is_within_alberta(lat, lng):
            position = [lat, lng]
            return position
        else:
            raise exceptions.PreventUpdate
    
    else:
        raise exceptions.PreventUpdate

@callback(
        Output(component_id="tooltip", component_property="children"), 
        Input(component_id="map", component_property="clickData"),
        Input('search-button', 'n_clicks'), 
        Input(component_id='store', component_property='data'),
        State('search-bar', 'value'),
        prevent_initial_call=True
        )
def show_tooltip_content(onClickData, n_clicks, store_data, search_value):
    print(store_data)
    
    if search_value is not None and search_value != "" and store_data['search'] != "":
        # position = getCoordinates(search_value)
        # if position != "":
        #     lat, lng = position
        lat, lng = store_data['search']
    elif onClickData is not None:
        lt, ln = Utils.get_latlng(onClickData)
        if is_within_alberta(lt, ln):
            lat, lng = lt, ln
        else:
            raise exceptions.PreventUpdate
    else:
        raise exceptions.PreventUpdate
    
    selected_model = model_loaders[store_data['model']]

    # TODO possible refactor
    if 'GP' in store_data['model']:
        means, stds = Utils.get_confidence_interval(lat, lng, data_loader, selected_model, number_of_stations)
        most_recent_mean = Utils.get_most_recent_model_prediction(means)
        most_recent_std = Utils.get_most_recent_model_prediction(stds)
        most_recent_wind_speed = str(np.round(most_recent_mean, 3)) + ' ± ' + str(np.round(most_recent_std, 3))
    # TS and NN cant provide confidence interval
    else:
        wind_speed = Utils.get_wind_speed(lat, lng, data_loader, selected_model, number_of_stations)
        most_recent_wind_speed = Utils.get_most_recent_model_prediction(wind_speed)

    #The DOM update is not triggered if the id stays the same
    #however, re-initialize the tooltip is necessary to change the content
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
        Input('search-button', 'n_clicks'), 
        State('search-bar', 'value'),
        )
def show_click_info(onClickData, n_clicks, search_value):
    if search_value is not None and search_value != "":
        position = getCoordinates(search_value)
        if position != "":
            lat, lng = position
            return f'Location clicked: {np.round(lat, 4)}, {np.round(lng, 4)}'
        else:
            return 'Location is not within Alberta. Try again.'
    elif onClickData is not None:
        lat, lng = Utils.get_latlng(onClickData)
        if is_within_alberta(lat, lng):
            return f'Location clicked: {np.round(lat, 4)}, {np.round(lng, 4)}'
        else:
            return 'Location is not within Alberta. Try again.'
    else:
        raise exceptions.PreventUpdate
    
@callback(
        Output(component_id="detail-graph", component_property="children"), 
        Input(component_id='map', component_property='clickData'),
        Input('search-button', 'n_clicks'), 
        State('search-bar', 'value'),
        Input(component_id='graph-dropdown', component_property='value'),
        )
def show_detail_graph(onClickData, n_clicks, search_value, value):
    if onClickData is None and (search_value is None or search_value == ""):
        raise exceptions.PreventUpdate
    else:
        return show_prediction_bound_plot(onClickData, search_value, value)

# TODO consider add a date slider to show the change of the prediction over time 
def show_prediction_bound_plot(onClickData, search_value, value):
    if onClickData is None and (search_value is None or search_value == ""):
        raise exceptions.PreventUpdate
    else:
        density = 10
        data = data_loader.get_data().sort_values('date', ascending=False).head(20*1)
        means, stds, latitudes, longitudes = Utils.get_grid_prediction(data, gp_martern_model_loader, density)

        if value == 'Lowerbound Prediction':
            bounds = np.maximum(0, np.subtract(means, np.multiply(1.96, stds)))
        elif value == 'Upperbound Prediction':
            bounds = np.add(means, np.multiply(1.96, stds))
            
        graph = get_graph(f'{str(value)} Contour Plot', latitudes, longitudes, bounds.reshape(density, density))
        return graph
    
@callback(
        Output(component_id="search-bar", component_property="value"),
        Input('search-button', 'n_clicks'),
                 
        prevent_initial_call=True
        )
def clearSearchBar(n_clicks):
    time.sleep(3)
    return ""
    
def is_coordinates(input_str):
    # Regular expression to check for coordinates pattern
    pattern = r'^-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+$'
    return re.match(pattern, input_str.strip()) is not None
def getCoordinates(search_value):
    position = ""
    if search_value:
        geolocator = Nominatim(user_agent="WindSpeedApproximationGeocoder")
        try:
            if is_coordinates(search_value):
                location = geolocator.reverse(search_value)
                if location:
                    lat, lng = map(float, search_value.split(','))
                    if is_within_alberta(lat, lng):
                        position = [lat, lng]
                        return position
                    else:
                        print("Location out of bounds.")
                        return position 
                else:
                    print("Location not found.")
                    return position
            else:
                location = geolocator.geocode(search_value)
                if location:
                    lat, lng = location.latitude, location.longitude
                    if is_within_alberta(lat, lng):
                        position = [lat, lng]
                        return position
                    else:
                        print("Location out of bounds.")
                        return position
                else:
                    print("Location not found.")
                    return position
        except GeopyError as e:
            print(f"Geocoding error: {e}")
            return position
    else:
        print('Please enter a location.')
        return position
def is_within_alberta(lat, lng):
    min_lat, min_lng = 49, -119
    max_lat, max_lng = 55, -109

    if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        return True
    else:
        return False

# UI elements
graph_section = html.Div(
    children=[map, graph], 
    style={
        'fontSize': '15px',
        'display': 'flex', 
        'flexDirection':'row', 
        'gap' : '10px',
        'padding-top': '10px',
        'backgroundColor': colors['background']
        }
    )

model_selection_section = html.Div(
    children=['Prediction Model: ', model_selector], 
    style={
        'fontSize': '15px',
        'display': 'flex', 
        'flexDirection':'row', 
        'align-items': 'center',
        'gap' : '10px',
        'padding-top': '10px',
        'backgroundColor': colors['background']
        }
    )

graph_selector_section = html.Div(
    children=['Detail Graph: ', graph_selector], 
    style={
        'display': 'flex', 
        'flexDirection':'row', 
        'align-items': 'center',
        'gap' : '10px',
        'padding-top': '10px',
        'backgroundColor': colors['background']
        }
    )

search_bar_section = html.Div(
    children=['Search: ', search_bar, search_button],
    style={
        'display': 'flex', 
        'flexDirection':'row', 
        'align-items': 'center',
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
        store,
        header,
        info,
        model_selection_section,
        graph_selector_section,
        search_bar_section,
        graph_section
])
app.run_server(debug=True)
 
