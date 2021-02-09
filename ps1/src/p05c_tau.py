import matplotlib.pyplot as plt
import numpy as np

import util
from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    def _MSE(x, y):
        return np.mean((x - y)**2)

    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # *** PART ONE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    MSE = []
    fig, axs = plt.subplots(2, 3, sharey=False)
    for ax, tau in zip(axs.flat, tau_values):
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)

        predictions = model.predict(x_eval)
        MSE.append(_MSE(predictions, y_eval))

        # Plotting
        ax.set_title(f"tau: {str(tau)}")
        ax.scatter(x_train[:, 1], y_train, color='blue', marker='.', alpha=0.5)
        ax.scatter(x_eval[:, 1],
                   predictions,
                   color='red',
                   marker='.',
                   alpha=0.5)
        ax.set_ylim(np.min(y_eval) / 2, np.max(y_eval) * 2)

    fig_path = pred_path[:-4] + "_eval_fig.jpg"
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    for tau, mse in zip(tau_values, MSE):
        print(f"MSE(tau = {tau}): {mse}")

    # *** PART TWO ***
    # Fit a LWR model with the best tau value
    tau = tau_values[MSE.index(min(MSE))]
    print("\ntau chosen:", tau)

    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)

    # Run on the test set to get the MSE value
    predictions = model.predict(x_test)
    MSE = _MSE(predictions, y_test)
    print(f"Test MSE(tau = {tau}): {mse}")

    # Save predictions to pred_path
    predictions = model.predict(x_test)
    np.savetxt(pred_path, predictions)
    # *** END CODE HERE ***
