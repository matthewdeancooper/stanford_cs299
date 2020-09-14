import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg

import util
from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)

    # Get MSE value on the validation set
    MSE = model.MSE(x_eval, y_eval, lwr=True)
    print("MSE:", MSE)

    # Plotting handled in next coding question
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def analytic_theta(self, prediction_instance, weight_matrix):
        transpose_x = self.x.transpose()
        inverse_product = linalg.inv(transpose_x @ weight_matrix @ self.x)
        return inverse_product @ transpose_x @ self.y

    def weight_matrix(self, prediction_instance):
        weight_matrix = np.zeros((self.x.shape[0], self.x.shape[0]))
        for i, train_instance in enumerate(self.x):
            weight_matrix[i, i] = 0.5 * self.weight_instance(
                train_instance, prediction_instance)
        return weight_matrix

    def weight_instance(self, train_instance, prediction_instance):
        weight_instance = np.exp(-((train_instance[1] - prediction_instance[1])**2) /
                                 (2 * self.tau**2))
        # assert weight_instance > 0
        return weight_instance

    def hypothesis(self, prediction_instance, theta):
        return theta @ prediction_instance

    def MSE(self, x, y, lwr=True):
        predictions = self.predict(x, lwr)
        return np.mean((predictions-y)**2)

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x, lwr=True):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        predictions = []
        for prediction_instance in x:
            if lwr:
                weight_matrix = self.weight_matrix(prediction_instance)
            else:
                weight_matrix = np.identity(self.x.shape[0])
            theta = self.analytic_theta(prediction_instance, weight_matrix)
            predictions.append(self.hypothesis(prediction_instance, theta))
        return np.array(predictions)
        # *** END CODE HERE ***
