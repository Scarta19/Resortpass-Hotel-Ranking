from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def train_model(df: pd.DataFrame):
    """
    Train a GradientBoostingRegressor model on the preprocessed data.
    """
    # Features
    X = df[['Price', 'Location_Proximity', 'Review_Score', 'Search_Count', 'CTR']]
    
    # Target (Bookings)
    y = df['Bookings']
    
    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the GradientBoostingRegressor model
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    
    # Return the trained model and test data for evaluation
    return model, X_test, y_test
