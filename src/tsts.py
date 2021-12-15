import numpy as np
from scipy.stats.stats import pearsonr
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

correl = 0.4


def generate_correlation():
    a = np.random.randint(0, 2, 100000, dtype='bool')

    b = a.copy()

    c = np.random.randint(0, 2, 100000, dtype='bool')

    d = a.copy()
    for i in range(d.shape[0]):
        if np.random.rand() > correl:
            d[i] = np.random.randint(0, 2, 1, dtype='bool')

    print(f'b: {pearsonr(a, b)[0]}')
    print(f'c: {pearsonr(a, c)[0]}')
    print(f'd: {pearsonr(a, d)[0]}')


def plot_sigmoid():
    x = np.linspace(-10, 10, 1000)
    y = 1 / (1 + np.exp(-x))

    plt.plot(x, y)
    plt.title('Sigmoid')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid()

    plt.show()


if __name__ == '__main__':
    plot_sigmoid()
