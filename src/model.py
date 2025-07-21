from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def train_model(X_train, y_train):
    """
    Train a Gradient Boosting model on the training data.
    """
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance on the test data.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_pred
