import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_top_10_hotels(predictions):
    """
    Visualize the top 10 hotels based on predicted bookings.
    """
    predictions_sorted = predictions.sort_values(by='Predicted_Bookings', ascending=False)
    top_10 = predictions_sorted.head(10)

    plt.figure(figsize=(10,6))
    sns.barplot(x='Predicted_Bookings', y='Hotel_ID', data=top_10, palette='viridis')
    plt.title('Top 10 Hotels Based on Predicted Bookings')
    plt.xlabel('Predicted Bookings')
    plt.ylabel('Hotel ID')
    plt.show()
