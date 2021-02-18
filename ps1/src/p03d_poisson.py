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
    # *** START CODE HERE ***
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    # Fit a Poisson Regression model
    theta_0 = np.zeros(x_train.shape[-1])

    model = PoissonRegression(theta_0=theta_0, step_size=lr)
    model.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    predictions = model.predict(x_eval)
    np.savetxt(pred_path, predictions)
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def grad_log_likelihood(self, x, y, cannonical_response):
        # Batch wise gradient
        gradient = np.zeros_like(self.theta)
        for j in range(len(self.theta)):
            grad_j = 0
            for x_i, y_i, response in zip(x, y, cannonical_response):
                grad_j += (y_i - response) * x_i[j]
            gradient[j] = grad_j
        return gradient / len(y)

    def natural_parameter(self, x):
        # Design choice to parameterise by a linear model
        return np.array([self.theta @ x_i for x_i in x])

    def cannonical_response(self, natural_parameter):
        # This is our hypothesis for Poisson
        # h(x) = E(y|x; theta) = (for poisson) lambda or exp(natural p)
        return np.exp(natural_parameter)

    # SGD
    # def grad_log_likelihood(self, x, y):
    #     # stochastic wise gradient
    #     # print("x", x, x.shape)
    #     # print("y", y, y.shape)
    #     n = self.theta @ x
    #     # print("n", n, n.shape)
    #     h = np.exp(n)
    #     # print("h",h, h.shape)
    #     grad = (y-h) * x
    #     # print("grad",grad, grad.shape)
    #     return grad

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # Batch gradient de
        iterations = 0
        while True:
            natural_parameter = self.natural_parameter(x)
            cannonical_response = self.cannonical_response(natural_parameter)
            grad_log_likelihood = self.grad_log_likelihood(
                x, y, cannonical_response)

            # Add update via gradient ascent
            theta_update = self.theta + self.step_size * grad_log_likelihood
            theta_difference = linalg.norm(theta_update - self.theta, ord=1)

            if theta_difference < self.eps:
                print(f"Converged after iterations: {iterations}")
                break
            else:
                self.theta = theta_update
                iterations += 1

        # SGD
        # theta_difference = self.eps + 1
        # while theta_difference > self.eps:
        #     for x_i, y_i in zip(x, y):
        #         theta_update = self.theta + self.step_size * self.grad_log_likelihood(x_i, y_i)
        #         theta_difference = linalg.norm(theta_update - self.theta, ord=1)
        #         print(theta_difference)
        #         self.theta = theta_update
        #         if theta_difference < self.eps:
        #             break
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


if __name__ == "__main__":
    print("\nTesting p03")
    main(lr=1e-7,
         train_path='../data/ds4_train.csv',
         eval_path='../data/ds4_valid.csv',
         pred_path='output/p03d_pred.txt')
