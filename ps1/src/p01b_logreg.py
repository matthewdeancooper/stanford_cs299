import numpy as np
from numpy import linalg

import util
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.
    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """

    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # Train a logistic regression classifier
    theta_0 = np.zeros(x_train.shape[-1])
    model = LogisticRegression(theta_0=theta_0)
    model.fit(x_train, y_train)

    # Use np.savetxt to save predictions on eval set to pred_path
    predictions = model.predict(x_eval)
    np.savetxt(pred_path, predictions)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.
    There are multiple ways to formulate this solution:
    1) Maximise the log likelihood by solving l'(theta) = 0 via Newtons method
    2) Minimise the cost J(theta) by solving J'(theta) = 0 via Newtons method
    3) Use gradient ascent to maximise the log likelihood l(theta)
    4) Use gradient descent to minimise J(theta)

    Here, we implement 2) and calculate the Hessian and gradients
    by batch gradient descent rather than stochastic gradient descent.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        Notes:
            theta: Shape (n, ).
        """

        # *** START CODE HERE ***
        def _hessian_log_liklihood(x):
            index_max = self.theta.shape[0]
            hessian = np.zeros((index_max, index_max))
            for k in range(index_max):
                for j in range(index_max):
                    H_kj = 0
                    for x_i in x:
                        H_kj += self.predict([x_i]) * (
                            1 - self.predict([x_i])) * x_i[j] * x_i[k]
                    hessian[k, j] = H_kj
            return hessian / len(x)

        def _grad_log_liklihood(x, y):
            gradient = np.zeros_like(self.theta)
            for j in range(len(gradient)):
                grad_j = 0
                for x_i, y_i in zip(x, y):
                    grad_j += (y_i - self.predict([x_i])) * x_i[j]
                gradient[j] = grad_j

            return -1 * gradient / len(x)

        iterations = 0
        while True:

            inverse_hessian = linalg.inv(_hessian_log_liklihood(x))
            theta_update = self.theta - inverse_hessian @ _grad_log_liklihood(
                x, y)
            theta_difference = linalg.norm(theta_update - self.theta, ord=1)

            if theta_difference < self.eps:
                print(f"Converged after iterations: {iterations}")
                break
            else:
                self.theta = theta_update
                iterations += 1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.
        Args:
            x: Inputs of shape (m, n). Assumes x[i][0] == 1
        Returns:
            Outputs of shape (m,).
        """

        # *** START CODE HERE ***
        def _sigmoid(z):
            return 1 / (1 + np.exp(-z))

        for x_i in x:
            assert x_i[0] == 1

        z = np.array([self.theta @ x_i for x_i in x])
        return _sigmoid(z)
        # *** END CODE HERE ***
