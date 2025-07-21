import pandas as pd
import numpy as np

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features and transforms existing ones to enhance the model.
    """
    # Creating price categories (Budget, Economy, Luxury)
    df['Price_Category'] = pd.cut(df['Price'], bins=[0, 200, 400, 500], labels=['Budget', 'Economy', 'Luxury'])
    
    # Review sentiment based on Review_Score (positive if score > 4.0 (basic logic), negative otherwise)
    df['Review_Sentiment'] = np.where(df['Review_Score'] > 4.0, 'Positive', 'Negative')
    
    return df
