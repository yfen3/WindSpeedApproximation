import pandas as pd

# A class that loads data from a file location
class DataLoader:
    def __init__(self, file_location=None):
        self.file_location = file_location
        self.data_frame = None
    
    def load_data_from_dataframe(self, dataframe):
        self.data_frame = dataframe

    def load_data_from_location(self):
        try:
            df = pd.read_csv(self.file_location)
            return df
        except FileNotFoundError:
            print("File not found.")
            return None
        
    def get_data(self):
        return self.data_frame