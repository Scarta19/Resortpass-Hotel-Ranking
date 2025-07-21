import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the hotel data by cleaning and scaling the features.
    """
    # Handling missing values
    df.fillna(df.mean(), inplace=True)
    
    # Scaling numerical features
    scaler = StandardScaler()
    df[['Price', 'Location_Proximity', 'Review_Score', 'Search_Count', 'CTR']] = scaler.fit_transform(
        df[['Price', 'Location_Proximity', 'Review_Score', 'Search_Count', 'CTR']]
    )
    
    return df
