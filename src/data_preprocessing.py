import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data() -> pd.DataFrame:
    """
    Loads the synthetic hotel data that we will generate.
    """
    # Generate the synthetic hotel data
    num_hotels = 1000
    hotel_data = {
        'Hotel_ID': [f'H{str(i).zfill(4)}' for i in range(1, num_hotels + 1)],
        'Price': np.random.randint(100, 500, num_hotels),  # Price between 100 and 500
        'Category': np.random.choice(['Luxury', 'Economy', 'Budget'], num_hotels),
        'Location_Proximity': np.random.uniform(0.5, 20.0, num_hotels),  # Distance in miles
        'Review_Score': np.random.uniform(2.0, 5.0, num_hotels),  # Rating between 1 and 5
        'Search_Count': np.random.randint(50, 1000, num_hotels),  # How many times this hotel was searched
        'CTR': np.random.uniform(0.05, 0.25, num_hotels),  # CTR between 5% and 25%
        'Bookings': np.random.randint(1, 300, num_hotels),  # Number of bookings
        'Time_Spent': np.random.uniform(30, 300, num_hotels),  # Time spent viewing in seconds
    }
    
    # Create a dataframe from the generated data
    return pd.DataFrame(hotel_data)

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the hotel data by handling missing values and scaling numerical features.
    """
    # Separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns

    # Handling missing values (fillna only for numeric columns)
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Scaling numerical features
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df

