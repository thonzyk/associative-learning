import numpy as np


def binary_cross_entropy(y_true, y_pred):
    return -((y_true * np.log(y_pred)) + ((1.0 - y_true) * np.log(1.0 - y_pred)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def linear_layer(x, w, b):
    return np.matmul(np.transpose(w), x) + b


def run_tst():
    y_pred = np.linspace(0.0, 1.0, 100)
    y_true = 1.0

    plt.figure()
    plt.plot(y_pred, binary_cross_entropy(y_true, y_pred))
    plt.title('binary_cross_entropy')

    x = np.linspace(-10.0, 10.0, 1000)
    plt.figure()
    plt.plot(x, sigmoid(x))
    plt.title('sigmoid')

    assert np.abs(
        linear_layer(np.array([1.0, 2.0, 3.0])[:, None], np.array([3.0, 2.0, 1.0])[:, None], 5.0) - 15.0) < 1e-6

    plt.show()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    run_tst()
