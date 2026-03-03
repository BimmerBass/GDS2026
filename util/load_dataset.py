import pandas as pd
import dagster as dg

class LoadDataConfig(dg.Config):
    filepath: str
    file_format: str

def load_data(config : LoadDataConfig) -> pd.DataFrame:
    pass