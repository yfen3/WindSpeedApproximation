import geopy.distance as distance
import numpy as np

# Get the lat and lng from the clicked location on the map
def get_latlng(onClickData):
    return onClickData['latlng']['lat'], onClickData['latlng']['lng']

def round_to_three_decimal_places(number):
    return np.round(number, 3)

def get_most_recent_model_prediction(model_prediction):
    return model_prediction[-1]

def get_model_prediction(scaled_input, model):
    model_prediction = model.predict(scaled_input)

    return model_prediction.flatten()

def get_scaled_input_to_predict(lat, lng, data_loader, number_of_stations, model_loader):
    model_input_data = generate_data(data_loader.get_data(), [lat, lng], number_of_stations)
    scaled_data = model_loader.load_standard_scaler(model_input_data)

    return scaled_data


def get_wind_speed(lat, lng, data_loader, model_loader, number_of_stations):
    input = get_scaled_input_to_predict(lat, lng, data_loader, number_of_stations, model_loader)

    wind_speed = get_model_prediction(input, model_loader.get_model())
    wind_speed = round_to_three_decimal_places(wind_speed)

    return wind_speed

def get_confidence_interval(lat, lng, data_loader, gp_model_loader, number_of_stations):
    model_input_data = generate_data(data_loader.get_data(), [lat, lng], number_of_stations)
    means, stds = gp_model_loader.get_model().predict(model_input_data, return_std=True)

    return means, stds

# generate a grid of points for visualization
# the geographical area is predefinded over the provence of Alberta
def generate_grid_points(density=10):
    longitudes = np.linspace(-110.07, -115.55, density)
    latitudes = np.linspace(49.12, 53.55, density)
    xx, yy = np.meshgrid(latitudes, longitudes)
    return np.array(list(zip(xx.flatten(), yy.flatten()))), latitudes, longitudes

# Retrun all prediction over the entire map by using the grid
# used to produce the contour plot of predictions
def get_grid_prediction(data, model_loader, density):
    grid_coor, latitudes, longitudes = generate_grid_points(density)

    grid_input_data = []
    for coor in grid_coor:
        input_feature = generate_data(data, coor, 9)
        if len(grid_input_data) == 0:
            grid_input_data = input_feature
        else:
            grid_input_data = np.concatenate((grid_input_data, input_feature), axis=0)

    means, stds = model_loader.get_model().predict(grid_input_data, return_std=True)

    return means, stds, latitudes, longitudes

def find_clostest_n_neighbours(target, unique_stations, number_of_neighbours):
    station_with_locations = unique_stations.copy()

    distances = station_with_locations.apply(
        lambda row: distance.distance(
            [row['latitude'], row['longitude']], [target[0], target[1]]).km,
        axis=1
    )
    station_with_locations['distance'] = distances

    station_in_range = station_with_locations.loc[(station_with_locations['distance'] >= 0)
                                                  & (station_with_locations['distance'] <= 99999)]

    station_to_use = station_in_range.nsmallest(number_of_neighbours, 'distance')
    
    return station_to_use


def extract_data(features, target_coor, features_to_use=None, target_features_to_use=None):
    if features_to_use is None:
        features_to_use = ['latitude', 'longitude', 'temp', 'wind_direction', 'wind_speed']
    if target_features_to_use is None:
        target_features_to_use = ['wind_speed']

    distances = features.apply(
        lambda row: distance.distance(
            [row['latitude'], row['longitude']], [target_coor[0], target_coor[1]]).km,
        axis=1
    )

    processed_features = features.loc[:,features_to_use].copy()
    processed_features['distance'] = distances

    processed_features = processed_features.to_numpy()

    return processed_features


# Note some station will have less data, so the smallest date range is used to mach all stations
def extract_data_match_date_range(features, target_coor, neighbour_station_names):
    processed_features = []

    for name in neighbour_station_names:
        selected_station_data = features.loc[features['name'] == name]
        extracted_features = extract_data(selected_station_data, target_coor)
        if len(processed_features) == 0:
            processed_features = extracted_features
        else:
            processed_features = np.concatenate((processed_features, extracted_features), axis=1)

    return processed_features    

# Given the target station name, find the nearest neighbours within the distance
def generate_data(raw_data, target_coor, number_of_neighbours):

    target_latitude = target_coor[0]
    target_longitude = target_coor[1]

    # select all unique names and coordinates
    unique_stations = raw_data.groupby('name').head(1)

    neighbour_stations = find_clostest_n_neighbours([target_latitude, target_longitude], unique_stations, number_of_neighbours)

    # find k nearest neighbours
    neighbour_station_names = neighbour_stations['name']

    # filter the data, return
    features = raw_data[raw_data['name'].isin(neighbour_station_names)]
    features = extract_data_match_date_range(features, target_coor, neighbour_station_names)

    return features