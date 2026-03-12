"""
AIstats_lab.py

Student starter file for the Regularization & Overfitting lab.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# =========================
# Helper Functions
# =========================

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# =========================
# Q1 Lasso Regression
# =========================

def lasso_regression_diabetes(lambda_reg=0.1, lr=0.01, epochs=2000):
    """
    Implement Lasso regression using gradient descent.
    """

    # Load dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Add bias column
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)

    # Initialize theta
    theta = np.zeros(X_train.shape[1])

    n = len(y_train)

    # Gradient Descent
    for _ in range(epochs):

        y_pred = X_train @ theta

        gradient = (2 / n) * (X_train.T @ (y_pred - y_train))

        # L1 Regularization (skip bias term)
        gradient[1:] += lambda_reg * np.sign(theta[1:])

        theta -= lr * gradient

    # Predictions
    train_pred = X_train @ theta
    test_pred = X_test @ theta

    # Metrics
    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q2 Polynomial Overfitting
# =========================

def polynomial_overfitting_experiment(max_degree=10):
    """
    Study overfitting using polynomial regression.
    """

    # Load dataset
    data = load_diabetes()
    X = data.data[:, 2].reshape(-1, 1)   # BMI feature only
    y = data.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    degrees = []
    train_errors = []
    test_errors = []

    # Loop through polynomial degrees
    for d in range(1, max_degree + 1):

        poly = PolynomialFeatures(degree=d)

        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Normal Equation
        theta = np.linalg.pinv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train

        train_pred = X_train_poly @ theta
        test_pred = X_test_poly @ theta

        train_errors.append(mse(y_train, train_pred))
        test_errors.append(mse(y_test, test_pred))
        degrees.append(d)

    return {
        "degrees": degrees,
        "train_mse": train_errors,
        "test_mse": test_errors
    }
