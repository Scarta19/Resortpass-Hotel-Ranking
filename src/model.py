from sklearn.ensemble import GradientBoostingRegressor

def train_model(df: pd.DataFrame) -> GradientBoostingRegressor:
    """
    Train a GradientBoostingRegressor model on the preprocessed data.
    """
    X = df[['Price', 'Location_Proximity', 'Review_Score', 'Search_Count', 'CTR']]  # Features
    y = df['Bookings']  # Target: number of bookings
    
    model = GradientBoostingRegressor()
    model.fit(X, y)
    
    return model
