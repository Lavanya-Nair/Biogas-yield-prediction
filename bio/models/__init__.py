from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_model(X, y):
    """
    Train an improved Random Forest model for biogas prediction
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    # Initialize model with tuned parameters
    model = RandomForestRegressor(
        n_estimators=500,  # Increased number of trees
        max_depth=15,      # Increased depth
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',  # Added feature selection
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    # Get feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    
    return {
        'model': model,
        'metrics': {
            'rmse': rmse,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        },
        'feature_importance': feature_importance,
        'test_data': (X_test, y_test, y_pred)
    }