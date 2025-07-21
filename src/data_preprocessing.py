import random
import pandas as pd
import numpy as np

def load_data():
    # Generate synthetic data for hotels
    num_hotels = 1000
    hotel_data = {
        'Hotel_ID': [f'H{str(i).zfill(4)}' for i in range(1, num_hotels + 1)],
        'Price': np.random.randint(100, 500, num_hotels),  # Price between 100 and 500
        'Category': np.random.choice(['Luxury', 'Economy', 'Budget'], num_hotels),
        'Location_Proximity': np.random.uniform(0.5, 20.0, num_hotels),  # Distance in miles
        'Review_Score': np.random.uniform(2.0, 5.0, num_hotels),  # Rating between 2 and 5
        'Amenities': [list(np.random.randint(0, 2, 5)) for _ in range(num_hotels)],  # List of 5 binary values for amenities
        'Search_Count': np.random.randint(50, 1000, num_hotels),  # How many times this hotel was searched
        'CTR': np.random.uniform(0.05, 0.25, num_hotels),  # CTR between 5% and 25%
        'Bookings': np.random.randint(1, 300, num_hotels),  # Random number of bookings
        'Time_Spent': np.random.uniform(30, 300, num_hotels),  # Time spent viewing the hotel
        'Review_Text': [f"Hotel {i} is a great choice for your stay. {random.choice(['Excellent', 'Good', 'Comfortable', 'Decent'])} amenities and service." for i in range(1, num_hotels + 1)]  # Example reviews
    }
    return pd.DataFrame(hotel_data)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the hotel data by handling missing values.
    """
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Handle missing values for numeric columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    return df
