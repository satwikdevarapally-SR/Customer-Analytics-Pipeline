import pandas as pd
import os

class DataIngestion:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path

    def load_data(self) -> pd.DataFrame:
        # Fail early if data file path doesn't exist
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Data file missing at: {self.raw_data_path}")
        
        return pd.read_csv(self.raw_data_path)