import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from DataLoader import DataLoader
from ModelLoader import ModelLoader

class AbWindDataModel:
    def __init__(self):
        self.df = pd.read_csv('../Data/processed_ab_wind_test.txt')
        self.nn_location = '../Visualization_proj_9stations.keras'

    def get_latest_one_month_records_data_loader(self):
        latest_one_month_records = self.df.sort_values('date', ascending=False).head(20*300)
        data_loader = DataLoader()
        data_loader.load_data_from_dataframe(latest_one_month_records)
        
        return data_loader


    def get_prediction_model(self):
        # Load the model
        nn_model = keras.models.load_model(self.nn_location)
        nn_model_loader = ModelLoader()
        nn_model_loader.load_model(nn_model)

        return nn_model_loader

