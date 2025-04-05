import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import xgboost as xgb

class CustomFocalLoss:
    """
    Custom Focal Loss implementation for regression problems
    
    Focal loss puts more focus on hard-to-predict samples by down-weighting easy examples.
    This implementation is adapted for regression problems.
    
    Parameters:
    -----------
    gamma : float
        Focusing parameter. Higher values put more focus on hard examples.
    alpha : float
        Balancing parameter. Controls the weight of samples.
    """
    
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
        self.name = "focal_regression_loss"
    
    def __call__(self, y_true, y_pred):
        """
        Calculate focal loss for regression
        
        Parameters:
        -----------
        y_true : tensor
            Ground truth values
        y_pred : tensor
            Predicted values
        
        Returns:
        --------
        tensor
            Calculated focal loss
        """
        # Calculate squared error (SE)
        se = K.square(y_true - y_pred)
        
        # Calculate modified focal weight
        pt = K.exp(-se)  # Higher for correct predictions
        focal_weight = K.pow(1 - pt, self.gamma)
        
        # Apply alpha weighting if needed
        if self.alpha is not None:
            # For regression, we can use different alphas for over/under predictions
            over_predicted = K.cast(K.greater(y_pred, y_true), K.floatx())
            alpha_weight = self.alpha * over_predicted + (1 - self.alpha) * (1 - over_predicted)
            focal_weight = focal_weight * alpha_weight
        
        # Final focal loss
        loss = focal_weight * se
        
        # Return mean loss
        return K.mean(loss)

def numpy_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    NumPy implementation of focal loss for regression
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values
    gamma : float
        Focusing parameter
    alpha : float
        Balancing parameter
    
    Returns:
    --------
    float
        Mean focal loss
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate squared error
    se = np.square(y_true - y_pred)
    
    # Calculate modified focal weight
    pt = np.exp(-se)  # Higher for correct predictions
    focal_weight = np.power(1 - pt, gamma)
    
    # Apply alpha weighting if needed
    if alpha is not None:
        over_predicted = (y_pred > y_true).astype(float)
        alpha_weight = alpha * over_predicted + (1 - alpha) * (1 - over_predicted)
        focal_weight = focal_weight * alpha_weight
    
    # Final focal loss
    loss = focal_weight * se
    
    # Return mean loss
    return np.mean(loss)

class XGBCustomFocalLossObjective:
    """
    Custom focal loss objective for XGBoost
    
    This class creates a custom objective function that can be used with XGBoost
    to implement focal loss for regression problems.
    
    Parameters:
    -----------
    gamma : float
        Focusing parameter
    alpha : float
        Balancing parameter
    """
    
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
    
    def __call__(self, predt, dtrain):
        """
        Calculate gradient and hessian for XGBoost's custom objective
        
        Parameters:
        -----------
        predt : numpy.ndarray
            Predicted values from XGBoost
        dtrain : xgboost.DMatrix
            Training data
        
        Returns:
        --------
        tuple
            Gradient and Hessian
        """
        y_true = dtrain.get_label()
        y_pred = predt
        
        # Calculate error
        error = y_pred - y_true
        
        # Calculate squared error for pt
        se = np.square(error)
        pt = np.exp(-se)
        
        # Calculate focal weight
        focal_weight = np.power(1 - pt, self.gamma)
        
        # Apply alpha weighting
        over_predicted = (error > 0).astype(float)
        alpha_weight = self.alpha * over_predicted + (1 - self.alpha) * (1 - over_predicted)
        focal_weight = focal_weight * alpha_weight
        
        # Gradient (first derivative)
        # d(focal_loss)/d(y_pred) = 2 * focal_weight * (y_pred - y_true)
        grad = 2 * focal_weight * error
        
        # Hessian (second derivative)
        # Simplified approximation: 2 * focal_weight
        hess = 2 * focal_weight  # Simplified
        
        return grad, hess

def train_xgboost_with_focal_loss(X_train, y_train, X_test, y_test, gamma=2.0, alpha=0.25, params=None):
    """
    Train XGBoost model with custom focal loss
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    X_test : array-like
        Test features
    y_test : array-like
        Test targets
    gamma : float
        Focusing parameter for focal loss
    alpha : float
        Balancing parameter for focal loss
    params : dict, optional
        Additional XGBoost parameters
    
    Returns:
    --------
    tuple
        (model, predictions, metrics)
    """
    # Create focal loss objective
    focal_obj = XGBCustomFocalLossObjective(gamma=gamma, alpha=alpha)
    
    # Default parameters if not provided
    if params is None:
        params = {
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_estimators': 200
        }
    
    # Create the model
    model = xgb.XGBRegressor(
        objective=focal_obj,
        **params
    )
    
    # Train the model
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=50,
        verbose=True,
        eval_metric=['rmse', 'mae']
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    
    # Calculate focal loss
    focal_loss_value = numpy_focal_loss(y_test, y_pred, gamma, alpha)
    metrics['Focal Loss'] = focal_loss_value
    
    print(f"XGBoost with Focal Loss (gamma={gamma}, alpha={alpha})")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return model, y_pred, metrics

# Example usage
if __name__ == "__main__":
    # Demo focal loss with random data
    print("Demo of focal loss with random data:")
    y_true = np.random.normal(5, 2, 1000)
    y_pred_perfect = y_true.copy()
    y_pred_close = y_true + np.random.normal(0, 0.5, 1000)
    y_pred_far = y_true + np.random.normal(0, 2, 1000)
    
    # Calculate loss for different prediction qualities
    fl = numpy_focal_loss(y_true, y_pred_perfect)
    print(f"Focal Loss (perfect): {fl:.6f}")
    
    fl = numpy_focal_loss(y_true, y_pred_close)
    print(f"Focal Loss (close): {fl:.6f}")
    
    fl = numpy_focal_loss(y_true, y_pred_far)
    print(f"Focal Loss (far): {fl:.6f}")
    
    # Calculate standard MSE for comparison
    mse_perfect = np.mean(np.square(y_true - y_pred_perfect))
    mse_close = np.mean(np.square(y_true - y_pred_close))
    mse_far = np.mean(np.square(y_true - y_pred_far))
    
    print(f"MSE (perfect): {mse_perfect:.6f}")
    print(f"MSE (close): {mse_close:.6f}")
    print(f"MSE (far): {mse_far:.6f}")
    
    print("Ratio of focal loss to MSE:")
    print(f"  Perfect: {fl/mse_perfect if mse_perfect > 0 else 'N/A'}")
    print(f"  Close: {fl/mse_close if mse_close > 0 else 'N/A'}")
    print(f"  Far: {fl/mse_far if mse_far > 0 else 'N/A'}")
    
    print("\nThe focal loss puts more weight on hard examples compared to MSE.") 