import pandas as pd
import os
from typing import Optional

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a dataset from the specified CSV file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset from {file_path} with {len(df)} rows and {len(df.columns)} columns.")
        return df
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        raise e

if __name__ == "__main__":
    # Test loading raw dataset if it exists
    raw_path = "data/raw/arxiv_raw.csv"
    if os.path.exists(raw_path):
        load_dataset(raw_path)
    else:
        print(f"Test run: Raw dataset does not exist yet at {raw_path}.")
