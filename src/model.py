from sklearn.ensemble import GradientBoostingRegressor

def train_model(X_train, y_train):
    """
    Trains the Gradient Boosting model.
    """
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    """
    Predicts with the trained model.
    """
    return model.predict(X_test)
