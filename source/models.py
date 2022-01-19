import numpy as np
from learning_functions import binary_cross_entropy, sigmoid, linear_layer


def dQ_dw(x, w, b):
    num = np.exp(np.matmul(np.transpose(w), x) + b)
    den = (np.exp(np.matmul(np.transpose(w), x) + b) + 1) ** 2
    return np.matmul(x, (num / den))


def dQ_db(x, w, b):
    num = np.exp(-np.matmul(np.transpose(w), x) - b)
    den = (np.exp(-np.matmul(np.transpose(w), x) - b) + 1) ** 2

    return num / den


def dR_dQ(x, w, b, y_true):
    q = Q(x, w, b)
    num = y_true - q
    den = np.matmul((1 - q), q)
    return num / den


def Q(x, w, b):
    return sigmoid(linear_layer(x, w, b))


class Perceptron(object):
    def __init__(self, num_of_inputs, learning_rate):
        self.num_of_inputs = num_of_inputs
        self.w = np.array([2.0, 10.0], dtype='float64')
        self.b = np.array([-20.0])

        # self.w = np.array([10.0, -10.0], dtype='float64')
        # self.b = np.array([-0.0])
        self.learning_rate = learning_rate

    def reset(self):
        self.w = np.random.normal(0.0, 1.0, self.num_of_inputs)
        self.b = np.random.normal(0.0, 1.0, 1)

    def predict(self, x):
        return sigmoid(linear_layer(x, self.w, self.b))

    def train_step(self, x, y_true):
        dq_dw = dQ_dw(x, self.w, self.b)
        dq_db = dQ_db(x, self.w, self.b)
        dr_dq = dR_dQ(x, self.w, self.b, y_true)

        w_update = dr_dq * dq_dw
        b_update = dr_dq * dq_db

        w_update *= self.learning_rate
        b_update *= self.learning_rate

        self.w += w_update
        self.b += b_update

    def eval(self, x, y_true):
        y_pred = sigmoid(linear_layer(x, self.w, self.b))
        return binary_cross_entropy(y_true, y_pred)

    def get_w(self):
        return self.w.copy()

    def get_b(self):
        return self.b.copy()

    def update_lr(self, lr):
        self.learning_rate = lr


def run_tst():
    perc = Perceptron(2, 0.1)
    print()


if __name__ == '__main__':
    run_tst()
