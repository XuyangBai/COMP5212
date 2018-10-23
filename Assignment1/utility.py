import numpy as np


def read_data(filename, standarize=True, verbose=False):
    """

    :rtype: object
    """
    with np.load(filename) as data:
        train_X = data['train_X']
        train_Y = data['train_Y']
        test_X = data['test_X']
        test_Y = data['test_Y']
        test_Y = test_Y.reshape(len(test_Y), 1)
        train_Y = train_Y.reshape(len(train_Y), 1)
        if standarize:
            train_X = (train_X - np.mean(train_X, axis=0)) / np.std(train_X, axis=0)
            test_X = (test_X - np.mean(test_X, axis=0)) / np.std(test_X, axis=0)
        if verbose:
            print("train_X shape:", train_X.shape)  # [n_samples, n_features]
            print("train_Y shape:", train_Y.shape)
        return train_X, train_Y, test_X, test_Y
