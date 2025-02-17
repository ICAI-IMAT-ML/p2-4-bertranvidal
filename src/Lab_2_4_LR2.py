import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session

        # Store the intercept and the coefficients of the model
        X_int = np.c_[np.ones(X.shape[0]), X]  # Concatenar una columna de 1s a X

        # Aplicar la ecuación normal: (X^T X)^(-1) X^T y
        matriz = np.linalg.inv(X_int.T @ X_int) @ X_int.T @ y

        # Extraer intercepto y coeficientes
        self.intercept = matriz[0]  # El primer valor es el intercepto
        self.coefficients = matriz[1:]  # El resto son los coeficientes

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01

        # Implement gradient descent (TODO)
        for epoch in range(iterations):
            predictions = self.predict(X)
            error = predictions - y

            # TODO: Write the gradient values and the updates for the paramenters
            intercept_gradient = np.mean(error)  # Gradiente del intercepto
            coefficient_gradient = (1/m) * (X[:, 1:].T @ error)  # Gradiente de los coeficientes

            self.intercept -= learning_rate * intercept_gradient
            self.coefficients -= learning_rate * coefficient_gradient


            # TODO: Calculate and print the loss every 10 epochs
            if epoch % 1000 == 0:
                mse  = np.sum(error**2)
                print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            # Predict when X is only one variable
            predictions = self.coefficients * X + self.intercept
        else:
            # Predict when X is more than one variable
            predictions = np.dot(X[:, 1:], self.coefficients) + self.intercept


        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    # TODO: Calculate R^2
    y_mean = np.mean(y_true)
    RSS = np.sum((y_true- y_pred)**2)
    TSS = np.sum((y_true- y_mean)**2)
    FUV = RSS/TSS

    r_squared = 1 - FUV

    # Root Mean Squared Error
    # TODO: Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))

    # Mean Absolute Error
    # TODO: Calculate MAE
    mae = np.mean(np.abs(y_true - y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}




def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = np.array(X, dtype=object)  # Asegurar que se trabaja con una copia

    for index in sorted(categorical_indices, reverse=True):  
        # Extraer la columna categórica
        categorical_column = X_transformed[:, index]

        # Encontrar valores únicos
        unique_values = np.unique(categorical_column)

        # Mapear cada categoría a un índice numérico
        category_to_index = {val: i for i, val in enumerate(unique_values)}
        numeric_column = np.array([category_to_index[val] for val in categorical_column])

        # Crear matriz One-Hot
        one_hot = np.eye(len(unique_values))[numeric_column]

        # Eliminar la primera columna si se solicita drop_first=True
        if drop_first:
            one_hot = one_hot[:, 1:]

        # Eliminar la columna categórica original y añadir la codificada al principio
        X_transformed = np.delete(X_transformed, index, axis=1)
        X_transformed = np.hstack([one_hot, X_transformed])  # One-Hot antes de las numéricas

    return X_transformed

