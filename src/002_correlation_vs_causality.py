import os
import shutil

import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.model_001 import get_binary_perceptron

matplotlib.use('Qt5Agg')


def create_data(n_of_samples=1000):
    x = np.random.randint(0, 2, 13 * n_of_samples, dtype='bool').reshape((n_of_samples, 13))

    for i in range(n_of_samples):
        if np.random.rand() > 0.5:
            x[i, 3] = x[i, 0]
        if np.random.rand() > 0.9:
            x[i, 4] = x[i, 0]
        if np.random.rand() > 0.99:
            x[i, 5] = x[i, 0]

        if np.random.rand() > 0.5:
            x[i, 6] = x[i, 1]
        if np.random.rand() > 0.9:
            x[i, 7] = x[i, 1]
        if np.random.rand() > 0.99:
            x[i, 8] = x[i, 1]

        if np.random.rand() > 0.5:
            x[i, 9] = x[i, 2]
        if np.random.rand() > 0.9:
            x[i, 10] = x[i, 2]
        if np.random.rand() > 0.99:
            x[i, 11] = x[i, 2]

    y = np.logical_and(np.logical_or(x[:, 0], x[:, 1]), np.logical_not(x[:, 2]))

    for i in range(n_of_samples):
        if np.random.rand() > 0.5:
            y[i] = np.random.randint(0, 2, dtype='bool')

    return x, y


def run():
    x, y = create_data()

    model = get_binary_perceptron(13, 1e0, True)

    model.fit(
        x, y,
        epochs=100,
        batch_size=100,
        shuffle=True,
        verbose=1,
    )

    print()


if __name__ == '__main__':
    run()
