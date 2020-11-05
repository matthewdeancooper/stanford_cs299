import numpy as np
from numpy import linalg

import util
from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    print(x_train.shape)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    theta_0 = np.zeros(x_train.shape[-1])
    model = PoissonRegression(theta_0=theta_0)
    model.fit(x_train, y_train)
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    # predictions = model.predict(x_eval)
    # np.savetxt(pred_path, predictions)
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def grad_log_likelihood(self, x, y, cannonical_response):
        # Batch wise
        gradient = np.zeros_like(self.theta)
        for j in range(len(self.theta)):
            grad_j = 0
            for x_i, y_i, response in zip(x, y, cannonical_response):
                grad_j += (y_i - response) * x_i[j]
            gradient[j] = grad_j
            print(grad_j)
        return gradient * (1/len(y))

    def natural_parameter(self, x):
        # Design choice to parameterise by a linear model
        return np.array([self.theta @ x_i for x_i in x])

    def cannonical_response(self, natural_parameter):
        # This is our hypothesis for Poisson
        return np.array([np.exp(n) for n in natural_parameter])

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        iterations = 0

        for i in range(2):

            natural_parameter = self.natural_parameter(x)
            cannonical_response = self.cannonical_response(natural_parameter)
            # Add update via gradient descent
            grad_log_likelihood = self.grad_log_likelihood(x, y, cannonical_response)
            theta_update = self.theta + self.step_size * grad_log_likelihood
            theta_difference = linalg.norm(theta_update - self.theta, ord=1)

            if theta_difference < self.eps:
                print(f"Converged after iterations: {iterations}")
                break
            else:
                self.theta = theta_update
                iterations += 1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***

        natural_parameter = self.natural_parameter(x)
        cannonical_response = self.cannonical_response(natural_parameter)
        return cannonical_response
        # *** END CODE HERE ***
