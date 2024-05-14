import geopy.distance as distance
import numpy as np

# Get the lat and lng from the clicked location on the map
def get_latlng(onClickData):
    return onClickData['latlng']['lat'], onClickData['latlng']['lng']

def get_model_prediction(scaled_input, model):
    model_prediction = model.predict(scaled_input)
    wind_speed = model_prediction[0][0]
    print(wind_speed)

    return wind_speed

def get_wind_speed(lat, lng, data_loader, model_loader, number_of_stations):
    model_input_data = generate_data(data_loader.get_data(), [lat, lng], number_of_stations)
    scaled_data = model_loader.load_standard_scaler(model_input_data)
    wind_speed = get_model_prediction(scaled_data, model_loader.get_model())

    return wind_speed

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