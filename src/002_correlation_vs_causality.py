import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.model_001 import get_binary_perceptron

matplotlib.use('Qt5Agg')


def create_data(n_of_samples=1000, noise_portion=0.0):
    x = np.random.randint(0, 2, 13 * n_of_samples, dtype='bool').reshape((n_of_samples, 13))

    for i in range(n_of_samples):
        if np.random.rand() < 0.1:
            x[i, 3] = x[i, 0]
        if np.random.rand() < 0.5:
            x[i, 4] = x[i, 0]
        if np.random.rand() < 0.9:
            x[i, 5] = x[i, 0]

        if np.random.rand() < 0.1:
            x[i, 6] = x[i, 1]
        if np.random.rand() < 0.5:
            x[i, 7] = x[i, 1]
        if np.random.rand() < 0.9:
            x[i, 8] = x[i, 1]

        if np.random.rand() < 0.1:
            x[i, 9] = x[i, 2]
        if np.random.rand() < 0.5:
            x[i, 10] = x[i, 2]
        if np.random.rand() < 0.9:
            x[i, 11] = x[i, 2]

    y = np.logical_and(np.logical_or(x[:, 0], x[:, 1]), np.logical_not(x[:, 2]))

    for i in range(n_of_samples):
        if np.random.rand() < noise_portion:
            y[i] = np.random.randint(0, 2, dtype='bool')

    return x, y


def run():
    x_train, y_train = create_data(20, 0.0)
    # x_train[:, 3:] = False
    x_val, y_val = create_data(1000, 0.0)

    model = get_binary_perceptron(13, 1e0, True)

    for i in range(20):
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=1,
            batch_size=100,
            shuffle=True,
            verbose=1,
        )

    plt.figure()
    plt.bar(['A', 'B', 'C', 'A: 0.5', 'A: 0.9', 'A: 0.99', 'B: 0.5', 'B: 0.9', 'B: 0.99', 'C: 0.5', 'C: 0.9', 'C: 0.99',
             'random'], model.get_weights()[0].flatten())

    plt.grid()
    plt.show()


if __name__ == '__main__':
    run()
