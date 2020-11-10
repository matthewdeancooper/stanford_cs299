import numpy as np
from numpy import linalg

import util
from linear_model import LinearModel
from p01b_logreg import LogisticRegression

import matplotlib.pyplot as plt


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # *** START CODE HERE ***
    # Train a GDA classifier
    # NOTE Drop x0 = 1 convention as in regression examples
    # Will need to account for this to write in terms of theta
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    model = GDA()
    model.fit(x_train, y_train)

    predictions = model.predict(x_eval)
    np.savetxt(pred_path, predictions)

    # Train Logistic regression classifier
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    theta_0 = np.zeros(x_train.shape[-1])
    model2 = LogisticRegression(theta_0=theta_0)
    model2.fit(x_train, y_train)

    # Plot decision boundary on validation set
    # Compare decision boundary with logistic
    thetas = [model.theta, model2.theta]
    fig_path = pred_path[:-4] + "_fig.jpg"
    colours = ["red", "orange"]
    util.plot_multiple(x_eval, y_eval, thetas, colours, fig_path)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def phi(self, y):
        return sum(y == 1) / len(y)

    def mu_0(self, x, y):
        return ((y == 0) @ x) / sum(y == 0)

    def mu_1(self, x, y):
        return ((y == 1) @ x) / sum(y == 1)

    def sigma(self, x, y, mu_0, mu_1):
        sigma = np.zeros((mu_0.shape[0], mu_0.shape[0]))
        for x_i, y_i in zip(x, y):
            if y_i == 0:
                mu = mu_0
            else:
                mu = mu_1
            x_i.shape = (len(x_i), 1)
            sigma += (x_i - mu) @ (x_i - mu).transpose()
        return sigma / len(y)

    def construct_theta(self, phi, mu_0, mu_1, sigma):
        inverse_sigma = linalg.inv(sigma)
        theta = inverse_sigma @ (mu_1 - mu_0)
        theta_0 = 0.5 * (mu_0 @ inverse_sigma @ mu_1 -
                         mu_1 @ inverse_sigma @ mu_0) - np.log((1 - phi) / phi)
        return np.append(theta_0, theta)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def probability(self, x):
        # Returns the probability of y=1|x in logistic form
        theta_0, theta = self.theta[0], self.theta[1:]
        # Add theta_0 to account for missing x_0 = 1's
        z = np.array([theta @ x_i for x_i in x]) + theta_0
        assert z.shape == (x.shape[0], )
        return self.sigmoid(z)

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma by maximising log likelihood
        phi = self.phi(y)
        mu_0 = self.mu_0(x, y)
        mu_1 = self.mu_1(x, y)
        sigma = self.sigma(x, y, mu_0, mu_1)
        # Write theta in terms of the parameters for linear decision boundary
        self.theta = self.construct_theta(phi, mu_0, mu_1, sigma)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        predictions = self.probability(x)
        return predictions
        # *** END CODE HERE
