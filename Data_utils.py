# The distiller class
# Modified from source: https://www.kaggle.com/code/lonnieqin/knowledge-distillation

import sklearn
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import geopy.distance as distance


# Source: https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
def find_clostest_n_neighbours(target, unique_stations, number_of_neighbours, max_threshold_distance,
                               min_threshold_distance):
    station_with_locations = unique_stations.copy()

    distances = station_with_locations.apply(
        lambda row: distance.distance(
            [row['latitude'], row['longitude']], [target[0], target[1]]).km,
        axis=1
    )
    station_with_locations['distance'] = distances

    station_in_range = station_with_locations.loc[(station_with_locations['distance'] >= min_threshold_distance)
                                                  & (station_with_locations['distance'] <= max_threshold_distance)
                                                  & (station_with_locations['distance'] != 0)]

    station_to_use = station_in_range.nsmallest(number_of_neighbours, 'distance')

    station_not_to_use = unique_stations.loc[~unique_stations['name'].isin(station_to_use['name'])]

    return station_to_use, station_not_to_use


def fill_missing_stations(features_by_date, station_not_to_use):
    new = station_not_to_use.copy()
    new.loc[:, ['date']] = features_by_date.iloc[0]['date']
    new.loc[:, ['temp']] = 0
    new.loc[:, ['precip']] = 0
    new.loc[:, ['wind_direction']] = 0
    new.loc[:, ['wind_speed']] = 0
    # New data as pandas.DataFrame
    new = pd.DataFrame(data=new)
    features_by_date = pd.concat([features_by_date, new])
    features_by_date.sort_values(by=['name'], inplace=True)

    return features_by_date


def extract_data(features, target, features_to_use=None, target_features_to_use=None):
    if features_to_use is None:
        features_to_use = ['latitude', 'longitude', 'temp', 'wind_direction', 'wind_speed']
    if target_features_to_use is None:
        target_features_to_use = ['wind_speed']

    distances = features.apply(
        lambda row: distance.distance(
            [row['latitude'], row['longitude']], [target['latitude'], target['longitude']]).km,
        axis=1
    )

    processed_features = features.loc[:,features_to_use].copy()
    processed_features['distance'] = distances

    processed_features = processed_features.to_numpy().flatten()

    processed_target = target.loc[target_features_to_use].copy()

    return processed_features, processed_target


# Note some station will have less data, so the smallest date range is used to mach all stations
def extract_data_match_date_range(features, target, neighbour_station_names, features_to_use, target_features_to_use):
    processed_features = []
    processed_target = []

    for index in range(len(target)):
        features_by_date = features.loc[features['date'] == target.iloc[index]['date']]
        if features_by_date.shape[0] == len(neighbour_station_names):
            extracted_features, extracted_target = extract_data(features_by_date, target.iloc[index], features_to_use, target_features_to_use)
            processed_features.append(extracted_features)
            processed_target.append(extracted_target)

    return processed_features, processed_target


# Given the target station name, find the nearest neighbours within the distance
def generate_data(raw_data, target_station_name, number_of_neighbours, max_threshold_distance,
                  min_threshold_distance=2, features_to_use=None, target_features_to_use=None):
    target = raw_data.loc[raw_data['name'] == target_station_name]

    target_latitude = target.iloc[0]['latitude']
    target_longitude = target.iloc[0]['longitude']

    # select all unique names and coordinates
    unique_stations = raw_data.groupby('name').head(1)

    neighbour_stations, station_not_to_use = find_clostest_n_neighbours([target_latitude, target_longitude],
                                                                        unique_stations, number_of_neighbours,
                                                                        max_threshold_distance, min_threshold_distance)

    if neighbour_stations.shape[0] != number_of_neighbours:
        print(
            f"target {target_station_name} has neighbour_stations {neighbour_stations.shape[0]} not match number_of_neighbours {number_of_neighbours}")

        features = []
        target = []
    else:
        # find k nearest neighbours
        neighbour_station_names = neighbour_stations['name']

        # filter the data, return
        features = raw_data[raw_data['name'].isin(neighbour_station_names)]
        features, target = extract_data_match_date_range(features, target, neighbour_station_names, features_to_use, target_features_to_use)

    return features, target
