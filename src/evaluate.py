import matplotlib.pyplot as plt
import seaborn as sns

def visualize_top_10_hotels(predictions):
    """
    Visualize the top 10 hotels based on predicted bookings.
    Ensures predicted bookings are integers for clarity.
    """
    # Sort the predictions by predicted bookings in descending order
    predictions_sorted = predictions.sort_values(by='Predicted_Bookings', ascending=False)
    
    # Take the top 10 hotels with the highest predicted bookings
    top_10 = predictions_sorted.head(10)

    # Plot the top 10 hotels based on predicted bookings
    plt.figure(figsize=(10,6))
    sns.barplot(x='Predicted_Bookings', y='Hotel_ID', data=top_10, palette='viridis', hue='Hotel_ID', legend=False)
    plt.title('Top 10 Hotels Based on Predicted Bookings')
    plt.xlabel('Predicted Bookings (Rounded)')
    plt.ylabel('Hotel ID')

    # Explicitly show the plot
    plt.show()
