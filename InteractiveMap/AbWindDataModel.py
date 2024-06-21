import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "torch"
import TsModel
import keras
from DataLoader import DataLoader
from ModelLoader import ModelLoader
import pickle

class AbWindDataModel:
    def __init__(self):
        self.df = pd.read_csv('../Data/processed_ab_wind_test.txt')
        self.nn_location = '../Visualization_proj_9stations.keras'

    def get_available_model_names():
        return {
            'NN': 'Neural Network',
            'TS' : 'TS fuzzy rule-based model',
            'GP (Martern)': 'Gaussian Process (Martern)',
            'GP TS (Martern)': 'Gaussian Process (Martern) with trained on TS prototypes'
        }

    def get_latest_one_month_records_data_loader(self):
        latest_one_month_records = self.df.sort_values('date', ascending=False).head(20*300)
        data_loader = DataLoader()
        data_loader.load_data_from_dataframe(latest_one_month_records)
        
        return data_loader


    def get_nn_model(self):
        # Load the model
        nn_model = keras.models.load_model(self.nn_location)
        nn_model_loader = ModelLoader()
        nn_model_loader.load_model(nn_model)

        return nn_model_loader
    
    def get_gp_martern_model(self):
        # Load the model
        with open('../gp_martern_model.pickle', 'rb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            gp_model = pickle.load(f)
        gp_model_loader = ModelLoader()
        gp_model_loader.load_model(gp_model)

        return gp_model_loader
    
    def get_gp_martern_ts_model(self):
        # Load the model
        with open('../gp_martern_ts_model.pickle', 'rb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            gp_model = pickle.load(f)
        gp_model_loader = ModelLoader()
        gp_model_loader.load_model(gp_model)

        return gp_model_loader
    
    def get_ts_model(self):
        # Load the model
        with open('../ts_model.pickle', 'rb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            ts_model = pickle.load(f)
        ts_model_loader = ModelLoader()
        ts_model_loader.load_model(ts_model)

        return ts_model_loader



