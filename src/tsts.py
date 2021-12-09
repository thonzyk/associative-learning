import numpy as np
from scipy.stats.stats import pearsonr


def generate_correlation():
    a = np.random.randint(0, 2, 1000000, dtype='bool')

    b = a.copy()

    c = np.random.randint(0, 2, 1000000, dtype='bool')

    d = a.copy()
    for i in range(d.shape[0]):
        if np.random.rand() < 0.1:
            d[i] = np.random.randint(0, 2, 1, dtype='bool')

    print(f'b: {pearsonr(a, b)[0]}')
    print(f'c: {pearsonr(a, c)[0]}')
    print(f'd: {pearsonr(a, d)[0]}')


if __name__ == '__main__':
    generate_correlation()
