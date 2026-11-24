import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    weights = weights.copy().reshape(-1, 1)  # Initialize weights to zeros
    if method == 'batch':
        batch_size = m
    elif method == 'stochastic':
        batch_size = 1
    else:
        pass

    for _ in range(n_epochs):
        for i in range(0, m, batch_size):
            X_b = X[i:i+batch_size]
            y_b = y[i:i+batch_size]
            y_pred = X_b @ weights 
            error = y_pred - y_b
            weights -= (learning_rate * 2 * (X_b.T @ error)) / len(X_b)

    return weights.flatten()
