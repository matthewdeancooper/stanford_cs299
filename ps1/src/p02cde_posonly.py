import numpy as np

import util
from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    x_train, t_train = util.load_dataset(train_path,
                                         label_col='t',
                                         add_intercept=True)

    _, y_train = util.load_dataset(train_path,
                                   label_col='y',
                                   add_intercept=True)

    x_valid, y_valid = util.load_dataset(train_path,
                                         label_col='y',
                                         add_intercept=True)

    x_test, t_test = util.load_dataset(test_path,
                                       label_col='t',
                                       add_intercept=True)

    theta_0 = np.zeros(x_train.shape[-1])

    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    # This is the ideal case where all t_i's are labelled
    # Hence we can train directly
    model = LogisticRegression(theta_0=theta_0)
    model.fit(x_train, t_train)

    theta_c = model.theta
    predictions_c = model.predict(x_test)
    np.savetxt(pred_path_c, predictions_c)

    # Part (d): Train on y-labels and test on true labels
    # Here t_i's are not available for training
    # Make sure to save outputs to pred_path_d
    model = LogisticRegression(theta_0=theta_0)
    model.fit(x_train, y_train)

    theta_d = model.theta
    predictions_d = model.predict(x_test)
    np.savetxt(pred_path_d, predictions_d)

    # Part (e): Apply correction factor using validation set and test on trues
    # Here t_i's are not available for training, we use the correction factor
    # to convert p(y|x) to p(t|x)
    x_valid_labelled = []
    for x, y in zip(x_valid, y_valid):
        if y == 1:
            x_valid_labelled.append(x)

    # For validation to calculate alpha
    predictions = model.predict(np.array(x_valid_labelled))
    alpha = (1 / len(predictions)) * np.sum(predictions)

    # For test to scale by alpha
    predictions_e = predictions_d / alpha
    np.savetxt(pred_path_e, predictions_e)

    # PLOTTING
    # # Plot and use np.savetxt to save outputs to pred_path_e
    # Part c
    print("Plotting c")
    title = "Trained on t directly"
    fig_path = pred_path_c[:-4] + "_fig.jpg"
    util.plot(x_test, t_test, theta_c, fig_path, title=title)

    # Part d
    print("Plotting d")
    title = "Trained on y - no correction for t"
    fig_path = pred_path_d[:-4] + "_fig.jpg"
    util.plot(x_test, t_test, theta_d, fig_path, title=title)

    # Part e
    print("Plotting e")
    title = "Trained on y, corrected to infer t via alpha"
    fig_path = pred_path_e[:-4] + "_fig.jpg"
    util.plot(x_test, t_test, theta_d, fig_path, correction=alpha, title=title)

    thetas = [theta_c, theta_d, theta_d]
    colours = ["red", "orange", "yellow"]
    fig_path = "output/p02_all" + "_fig.jpg"
    corrections = [1, 1, alpha]
    util.plot_multiple(x_test, t_test, thetas, colours, fig_path, corrections)
    # *** END CODER HERE


if __name__ == "__main__":
    print("\nTesting p02")
    main(train_path='../data/ds3_train.csv',
         valid_path='../data/ds3_valid.csv',
         test_path='../data/ds3_test.csv',
         pred_path='output/p02X_pred.txt')
