import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    error = ((X @ w - y_true) ** 2).mean()
    ridge = sum(alpha * (w ** 2))
    return error + ridge
