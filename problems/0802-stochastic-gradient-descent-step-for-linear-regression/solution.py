import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n_samples = X.shape[0]
    for i in range(n_iter):
        size = i % n_samples
        x_i = X[size]
        y_i = y[size] 
        y_i_pred = x_i @ weights
        error_i = y_i_pred - y_i
        weights -= 2 * learning_rate * error_i * x_i
    return weights.tolist()
