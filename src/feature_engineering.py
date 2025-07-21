import pandas as pd
from sklearn.preprocessing import StandardScaler

def one_hot_encode(df):
    """
    One-hot encodes categorical columns in the dataframe.
    """
    df = pd.get_dummies(df, drop_first=True)
    return df

def scale_features(df):
    """
    Scales numerical features.
    """
    scaler = StandardScaler()
    df[['Price', 'Location_Proximity', 'Time_Spent']] = scaler.fit_transform(df[['Price', 'Location_Proximity', 'Time_Spent']])
    return df
