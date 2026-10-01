"""Linear regression from the Zoomcamp lessons, with optional L2 regularization."""

import numpy as np


def train_linear_regression(X: np.ndarray, y: np.ndarray):
    """Fit an unregularized model. Returns (bias, weights)."""
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


def train_linear_regression_reg(X: np.ndarray, y: np.ndarray, r: float = 0.0):
    """Fit a model with L2 penalty r added to the normal equation.

    The penalty is applied to every column of XTX, including the bias column,
    which is how the lecture implements regularization.
    """
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


def predict(X: np.ndarray, w0: float, w: np.ndarray) -> np.ndarray:
    return w0 + X.dot(w)


def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    error = y_pred - y
    mse = (error ** 2).mean()
    return float(np.sqrt(mse))
