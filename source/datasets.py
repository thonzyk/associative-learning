import numpy as np


def create_data():
    x_train = np.array([
        [1, 1],
        [0, 0]
    ], dtype='float64')

    y_train = np.array([
        1,
        0
    ], dtype='float64')

    x_test = np.array([
        [1, 1],
        [0, 1],
        [0, 0],
    ], dtype='float64')

    y_test = np.array([
        1,
        0,
        0
    ], dtype='float64')

    return x_train, y_train, x_test, y_test
