def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features and transforms existing ones to enhance the model.
    """
    # Creating price categories
    df['Price_Category'] = pd.cut(df['Price'], bins=[0, 200, 400, 500], labels=['Budget', 'Economy', 'Luxury'])
    
    # Review sentiment analysis (basic placeholder logic)
    df['Review_Sentiment'] = np.where(df['Review_Score'] > 4.0, 'Positive', 'Negative')
    
    return df
