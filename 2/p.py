import numpy as np

if __name__ == '__main__':
    a = np.array([1.0, 2.0, 3.0])[:, None]
    b = np.array([3.0, 2.0, 1.0])[None, :]

    c = b * a
    print()
