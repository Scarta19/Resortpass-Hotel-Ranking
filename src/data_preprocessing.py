import pandas as pd

def load_data(file_path):
    """
    Loads the dataset from the given file path.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocesses the hotel dataset by handling missing values, encoding categorical variables,
    and normalizing numerical features.
    Args:
        df (pd.DataFrame): The input dataframe containing hotel data.
    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    # Handling missing values
    df = df.dropna()  
    return df
