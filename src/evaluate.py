from sklearn.metrics import mean_squared_error

def evaluate_model(y_test, y_pred):
    """
    Evaluates the model using Mean Squared Error (MSE).
    """
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error (MSE): {mse}')
