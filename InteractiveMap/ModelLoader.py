import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load a ML model to do the prediction
# also it can return a data scaler to preprocess the data
class ModelLoader:
    def __init__(self):
        self.model = None
        self.scaler = None

    def load_model(self, model):
        self.model = model

    def load_standard_scaler(self, data):
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data

    def get_model(self):
        return self.model
    
    def get_standard_scaler(self):   
        return self.scaler