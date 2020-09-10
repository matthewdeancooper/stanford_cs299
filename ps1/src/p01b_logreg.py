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

    # Plot decision boundary on top of validation set
    fig_path = pred_path + "_fig.jpg"
    util.plot(x_eval, y_eval, model.theta, fig_path)

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

    Here, we implement 1) and calculate the Hessian and gradients accordingly.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def assert_x0(self, x):
        for x_i in x:
            assert x_i[0] == 1

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def hypothesis(self, x):
        # Using the vectored logistic/sigmoid function as the activation.
        # As theta has shape (n,) the transpose is an identity operation
        # @ is equivalent to the dot product for shapes (n,) @ (n,)
        if len(x.shape) == 1:
            z = self.theta @ x
        else:
            z = np.array([self.theta @ x_i for x_i in x])
            assert z.shape == (x.shape[0], )
        return self.sigmoid(z)

    def hessian_log_liklihood(self, x):
        hessian = np.zeros((self.theta.shape[0], self.theta.shape[0]))
        for k in range(len(self.theta)):
            for j in range(len(self.theta)):
                H_kj = 0
                for x_i in x:
                    H_kj += -self.hypothesis(x_i) * (
                        1 - self.hypothesis(x_i)) * x_i[j] * x_i[k]
                hessian[k, j] = H_kj
        assert hessian.shape == (self.theta.shape[0], self.theta.shape[0])
        return hessian

    def grad_log_liklihood(self, x, y):
        gradient = np.zeros_like(self.theta)
        for j in range(len(self.theta)):
            grad_j = 0
            for x_i, y_i in zip(x, y):
                grad_j += (y_i - self.hypothesis(x_i)) * x_i[j]
            gradient[j] = grad_j

        assert gradient.shape == self.theta.shape
        return gradient

    def newtons_method_step_size(self, x, y):
        inverse_hessian = linalg.inv(self.hessian_log_liklihood(x))
        return -inverse_hessian @ self.grad_log_liklihood(x, y)

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Notes:
            theta: Shape (n, ).
        """

        # *** START CODE HERE ***
        self.assert_x0(x)
        iterations = 0
        while True:

            # Add update and let step function handle direction
            theta_update = self.theta + self.newtons_method_step_size(x, y)
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
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        self.assert_x0(x)
        return self.hypothesis(x)
        # *** END CODE HERE ***
