from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

def tune_hyperparameters(X_train, y_train, model_type='grid', n_iter=100):
    """
    Tune hyperparameters using GridSearchCV or RandomizedSearchCV.
    
    Parameters:
    X_train (DataFrame): Training feature set.
    y_train (Series): Training target set.
    model_type (str): Type of search ('grid' for GridSearchCV, 'random' for RandomizedSearchCV).
    n_iter (int): Number of iterations for RandomizedSearchCV.
    
    Returns:
    best_model (estimator): The best model after hyperparameter tuning.
    best_params (dict): The best hyperparameters.
    """
    
    # Arbitrary grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10]
    }

    # Choose the type of search
    if model_type == 'grid':
        search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                             param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    elif model_type == 'random':
        search = RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                                    param_distributions=param_grid, n_iter=n_iter, cv=3, n_jobs=-1, verbose=2)

    # Fit the model with tthe best hyperparameters
    search.fit(X_train, y_train)
    
    # Return the best model and best hyperparameters
    best_model = search.best_estimator_
    best_params = search.best_params_
    
    return best_model, best_params
