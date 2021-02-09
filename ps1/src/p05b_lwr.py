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
    def _MSE(x, y):
        return np.mean((x - y)**2)

    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)

    # Get MSE value on the validation set
    predictions = model.predict(x_eval)
    MSE = _MSE(predictions, y_eval)
    print(f"MSE(tau = {tau}): {MSE}")

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

        def _weight_instance(train_instance, prediction_instance):
            weight_instance = np.exp(
                -((train_instance[1] - prediction_instance[1])**2) /
                (2 * self.tau**2))
            return weight_instance

        def _weight_matrix(prediction_instance):
            weight_matrix = np.zeros((self.x.shape[0], self.x.shape[0]))
            for i, train_instance in enumerate(self.x):
                weight_instance = _weight_instance(train_instance,
                                                   prediction_instance)
                weight_matrix[i, i] = 0.5 * weight_instance
            return weight_matrix

        def _analytic_theta(prediction_instance, weight_matrix):
            transpose_x = self.x.transpose()
            inverse_product = linalg.inv(transpose_x @ weight_matrix @ self.x)
            return inverse_product @ transpose_x @ self.y

        predictions = []
        for prediction_instance in x:
            if lwr:
                weight_matrix = _weight_matrix(prediction_instance)
            else:
                weight_matrix = np.identity(self.x.shape[0])

            analytic_theta = _analytic_theta(prediction_instance,
                                             weight_matrix)
            hypothesis = analytic_theta @ prediction_instance

            predictions.append(hypothesis)

        return np.array(predictions)
        # *** END CODE HERE ***
