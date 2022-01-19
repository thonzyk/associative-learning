from models import Perceptron
from datasets import create_data
from matplotlib import pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')

REPS = 10
EPOCHS = 100


def classical_conditioning_experiment(noise=0.0):
    # Initialize
    perc = Perceptron(2, 1e0)
    x_train, y_train, x_test, y_test = create_data()
    pred_history = [[], [], [], []]
    weights_history = []
    bias_history = []

    def run_train(x, y):
        for epoch in range(EPOCHS):
            if epoch > 3:
                weights_history.append(perc.get_w())
                bias_history.append(perc.get_b())
                for i in range(x_test.shape[0]):
                    pred_history[i].append(perc.predict(x_test[i, :][:, None]))

            for i in range(x.shape[0]):
                if np.random.rand() < noise:
                    perc.train_step(x[i, :][:, None], 1.0 - y[i])
                else:
                    perc.train_step(x[i, :][:, None], y[i])

    # Pairing 1
    run_train(x_train, y_train)
    # Recovery
    run_train(x_test, y_test)
    # Pairing 2
    perc.update_lr(1e2)
    run_train(x_train, y_train)

    plt.figure()
    for i in range(x_test.shape[0]):
        if i % 2 == 0:
            plt.plot(pred_history[i])
        else:
            plt.plot(pred_history[i])

    plt.vlines([(i + 1) * EPOCHS for i in range(2)], 0, 1, linestyle=':', color='black')
    plt.legend(['Input 1 1', 'Input 0 1', 'Input 0 0', 'Time of change'])
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Prediction')
    plt.title('Prediction development')

    plt.figure()
    plt.plot([el[0] for el in weights_history])
    plt.plot([el[1] for el in weights_history])
    plt.vlines([(i + 1) * EPOCHS for i in range(2)], np.min(np.concatenate(weights_history)),
               np.max(np.concatenate(weights_history)), linestyle=':', color='black')
    plt.title('Weights development')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Weight value')
    plt.legend(['Weight A', 'Weight B', 'Time of change'])

    plt.figure()
    plt.plot(bias_history)
    plt.vlines([(i + 1) * EPOCHS for i in range(2)], np.min(np.concatenate(bias_history)),
               np.max(np.concatenate(bias_history)), linestyle=':', color='black')

    plt.legend(['Bias', 'Time of change'])
    plt.title('Bias development')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Bias value')

    plt.show()


if __name__ == '__main__':
    classical_conditioning_experiment()
