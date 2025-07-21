import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features and transforms existing ones to enhance the model.
    """
    # One-hot encoding for categorical features (e.g., Category)
    df = pd.get_dummies(df, columns=['Category'], drop_first=True)

    # Interaction feature: Price * Location Proximity
    df['Price_Location_Interaction'] = df['Price'] * df['Location_Proximity']

    # TF-IDF for text features (Review_Text)
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    X_text = tfidf.fit_transform(df['Review_Text'])

    # Apply Truncated SVD (LSA) for dimensionality reduction
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_text_reduced = svd.fit_transform(X_text)

    # Convert reduced text features into a DataFrame
    text_features_df = pd.DataFrame(X_text_reduced, columns=[f'text_feature_{i+1}' for i in range(X_text_reduced.shape[1])])

    # Combine the text features with the other hotel features
    df = pd.concat([df, text_features_df], axis=1)

    # Drop the original 'Review_Text' column as it's now represented by the text features
    df.drop(columns=['Review_Text'], inplace=True)

    # Flatten the Amenities column by creating separate binary columns for each amenity
    amenities_df = pd.DataFrame(df['Amenities'].to_list(), columns=[f'amenity_{i+1}' for i in range(5)])

    # Drop the original 'Amenities' column
    df = pd.concat([df, amenities_df], axis=1)
    df.drop(columns=['Amenities'], inplace=True)

    return df
