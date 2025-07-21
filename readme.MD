# ResortPass Hotel Ranking System

This repository contains the implementation of an optimized hotel ranking system for ResortPass. The goal of the project is to improve the way hotels are ranked on the search results page (SRP) based on various criteria, using machine learning models and data science techniques. The model was developed using **Gradient Boosting Regressor**, and the ranking algorithm was fine-tuned to prioritize hotels based on predicted bookings.

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Modeling Approach](#modeling-approach)
- [Success Metrics](#success-metrics)
- [Challenges and Pitfalls](#challenges-and-pitfalls)
- [How to Use](#how-to-use)
- [Requirements](#requirements)
- [Conclusion](#conclusion)

## Overview

The current ResortPass hotel ranking system is manual and based on predefined criteria. In this project, I propose a data-driven solution using **Gradient Boosting** to automatically rank hotels based on features such as price, category, location, and user behavior data.

Key features of the project:
- Generation of **synthetic hotel data** to simulate real-world scenarios.
- **Model development** using Gradient Boosting Regressor.
- **Hyperparameter tuning** using GridSearchCV for model optimization.
- **Visualization** of the top 10 hotels based on predicted bookings.

## Data

The data used for this project is synthetic and generated to emulate real hotel data. It contains the following features:

- **Hotel_ID**: Unique identifier for each hotel.
- **Price**: Randomized price range for hotels (between $100 and $500).
- **Category**: Categorical variable indicating the type of hotel (Luxury, Economy, or Budget).
- **Location_Proximity**: Distance of the hotel from popular landmarks in miles.
- **Review_Score**: Randomized ratings between 2 and 5 for each hotel.
- **Amenities**: Binary features indicating the availability of amenities such as pool, gym, and Wi-Fi.
- **Search_Count**: Number of times a hotel has been searched.
- **CTR (Click-Through Rate)**: Percentage of times the hotel was clicked on after being shown in search results.
- **Bookings**: The number of bookings the hotel received (target variable).
- **Review_Text**: A synthetic text feature representing hotel reviews.

These features were engineered further through one-hot encoding for categorical variables, interaction terms like `Price_Location_Interaction`, and dimensionality reduction using **TruncatedSVD** for text features.

### **What Data Would Be Useful to Evaluate and Improve the Ranking Algorithm?**
To enhance the ranking algorithm further, the following data would be crucial:
- **User behavior data**: Click-through rates (CTR), conversion rates, and search history.
- **Hotel characteristics**: In-depth features such as availability, seasonal offers, and average customer satisfaction.
- **Geospatial data**: Proximity to tourist attractions, local events, and high-demand areas.
- **User reviews and sentiment analysis**: Use sentiment analysis to classify reviews into positive, neutral, and negative, which can help rank based on user sentiment.
  
This data allows for more personalized rankings based on individual preferences, improving user experience.

## Modeling Approach

### **Model Chosen**: **Gradient Boosting Regressor**
We selected **Gradient Boosting** due to its ability to handle both numerical and categorical data and model non-linear relationships. The model was trained on various hotel features and hyperparameters were tuned using **GridSearchCV** to optimize performance.

### **Steps in Modeling**:
1. **Feature Engineering**: Includes creating interaction terms (e.g., `Price_Location_Interaction`), encoding categorical variables using one-hot encoding, and processing text features using **TF-IDF** and **TruncatedSVD** for dimensionality reduction.
2. **Model Training**: The **GradientBoostingRegressor** model was trained using **train-test split** and **cross-validation**.
3. **Hyperparameter Tuning**: Used **GridSearchCV** to tune parameters like `n_estimators`, `learning_rate`, `max_depth`, and `subsample`.
4. **Evaluation**: Metrics like **Mean Squared Error (MSE)** and **NDCG** (Normalized Discounted Cumulative Gain) were used to evaluate the model's performance.

### **Hyperparameter Tuning**:
The hyperparameters tuned using **GridSearchCV** include:
- **n_estimators**: Number of boosting stages.
- **learning_rate**: Controls the contribution of each base learner.
- **max_depth**: Limits the depth of the trees to prevent overfitting.
- **subsample**: Fraction of samples to fit each tree to reduce variance.

## Success Metrics

The following **Key Performance Indicators (KPIs)** were tracked to measure the success of the ranking system:

- **Click-Through Rate (CTR)**: Measures how many users click on a hotel after seeing it in search results.
- **Conversion Rate**: Measures how many clicks lead to actual bookings.
- **Mean Squared Error (MSE)**: To assess the model's accuracy in predicting bookings.
- **NDCG (Normalized Discounted Cumulative Gain)**: Measures the quality of the ranking by giving higher importance to higher ranks.
- **Precision at K**: Measures the proportion of relevant hotels in the top `K` predictions.

### **Challenges**:
1. **Data Quality**: Handling missing or noisy data is essential to prevent model bias and inaccuracies.
2. **Model Overfitting**: It's important to avoid overfitting the model with too many features or complex hyperparameters. Regularization techniques and cross-validation can help prevent this.
3. **Scalability**: The model may be computationally expensive, especially for larger datasets. Optimizing training time and memory usage is essential.
4. **Bias**: Ensuring the model doesn’t unfairly favor popular hotels or specific categories, leading to less diversity in recommendations.

## How to Use

To use the model, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Scarta19/Resortpass-Hotel-Ranking
2. Install the required dependencies:
  pip install -r requirements.txt
3. Open the Jupyter Notebook:
4. Run the Notebook Cells

## Conclusion

This project provides a comprehensive solution to optimize the hotel ranking system on ResortPass using machine learning. The ranking algorithm is based on **Gradient Boosting** and is optimized using **hyperparameter tuning**. By implementing this model, ResortPass can improve the relevance of hotel search results and enhance user experience, ultimately boosting bookings and customer satisfaction.

### Next Steps
1. **Deployment**: The next phase would involve deploying this model into a live production environment, where real-time hotel booking data can be processed and ranked.
2. **A/B Testing**: We recommend conducting A/B tests to compare the performance of this new ranking algorithm against the current manual ranking system.
3. **Real Data**: While synthetic data was used for this project, implementing the model on real-world data can significantly improve the model’s effectiveness and precision.

This approach not only addresses the current challenge of manual ranking but also sets the foundation for continuous improvement in hotel ranking algorithms by leveraging machine learning techniques.

---

Thank you for reviewing the implementation. Feel free to reach out for any questions or further comments!
