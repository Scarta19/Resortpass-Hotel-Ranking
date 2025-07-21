from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using MSE and R-squared.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate MSE and R2 score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'MSE': mse, 'R-squared': r2}
