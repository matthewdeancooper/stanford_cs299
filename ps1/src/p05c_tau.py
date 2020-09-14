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
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    MSE = []
    fig, axs = plt.subplots(2, 3, sharey=False)
    for ax, tau in zip(axs.flat, tau_values):
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        MSE.append(model.MSE(x_eval, y_eval, lwr=True))
        predictions = model.predict(x_eval)

        # Plotting
        ax.set_title(f"tau: {str(tau)}")
        ax.scatter(x_train[:,1], y_train, color='blue', marker='.', alpha=0.5)
        ax.scatter(x_eval[:,1], predictions, color='red', marker='.', alpha=0.5)
        ax.set_ylim(np.min(y_eval)/2, np.max(y_eval)*2)

    fig_path = pred_path[:-4] + "_eval_fig.jpg"
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    for tau, mse in zip(tau_values, MSE):
        print("tau:", tau, ",", "MSE:", mse)

    # Fit a LWR model with the best tau value
    tau = tau_values[MSE.index(min(MSE))]
    print("tau chosen:", tau)

    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)

    # Run on the test set to get the MSE value
    MSE = model.MSE(x_test, y_test, lwr=True)
    print("MSE:", MSE)

    # Save predictions to pred_path
    predictions = model.predict(x_test)
    np.savetxt(pred_path, predictions)

    # Plot data
    fig_path = pred_path[:-4] + "_test_fig.jpg"
    plt.scatter(x_test[:,1], y_test)
    plt.scatter(x_test[:,1], predictions, color='red')
    plt.savefig(fig_path)
    plt.close()
    # *** END CODE HERE ***
