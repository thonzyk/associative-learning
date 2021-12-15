from modules.model_001 import get_binary_perceptron
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from tensorflow.keras.callbacks import ModelCheckpoint
import os, shutil
from tqdm import tqdm
import pandas as pd

matplotlib.use('Qt5Agg')

EPOCHS = 30
REPEATS = 1000


def clean_directory(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def create_data():
    x_train = np.array([
        [1, 1],
        [0, 0]
    ], dtype='bool')

    y_train = np.array([
        1,
        0
    ], dtype='bool')

    x_test = np.array([
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
    ], dtype='bool')

    y_test = np.array([
        1,
        1,
        0,
        0
    ], dtype='bool')

    return x_train, y_train, x_test, y_test


def vectorize(list_of_lists):
    if type(list_of_lists[0][0]) == np.ndarray:
        list_of_lists = [np.concatenate(el, axis=-1)[None, :] for el in list_of_lists]
    else:
        list_of_lists = [np.array(el)[None, :] for el in list_of_lists]
    list_of_lists = np.concatenate(list_of_lists, axis=0)
    return list_of_lists


def run_experiment(ratio):
    WEIGHT_A = []
    WEIGHT_B = []
    BIAS = []
    PREDS = []

    model = get_binary_perceptron(2, 1e0, True)

    x_train, y_train, x_test, y_test = create_data()

    for i in tqdm(range(REPEATS)):
        model.set_weights([np.random.randn(2).reshape((2, 1)), np.zeros((1,))])

        weight_a = []
        weight_b = []
        bias = []
        preds = []

        pred = model.predict(x_test)

        weights = model.get_weights()
        weight_a.append(weights[0][0])
        weight_b.append(weights[0][1])
        bias.append(weights[1][0])
        preds.append(pred)

        for epoch in range(EPOCHS):
            model.fit(
                x_train, y_train,
                validation_data=(x_test, y_test),
                epochs=1,
                batch_size=2,
                shuffle=True,
                verbose=0,
                class_weight={0: 1.0 / ratio, 1: 1.0}
            )

            pred = model.predict(x_test)
            weights = model.get_weights()
            weight_a.append(weights[0][0])
            weight_b.append(weights[0][1])
            bias.append(weights[1][0])
            preds.append(pred)

        WEIGHT_A.append(weight_a)
        WEIGHT_B.append(weight_b)
        BIAS.append(bias)
        PREDS.append(preds)

    WEIGHT_A = vectorize(WEIGHT_A)
    WEIGHT_B = vectorize(WEIGHT_B)
    BIAS = vectorize(BIAS)
    PREDS = vectorize(PREDS)

    avg_WEIGHT_A = np.mean(WEIGHT_A, axis=0)
    avg_WEIGHT_B = np.mean(WEIGHT_B, axis=0)
    avg_BIAS = np.mean(BIAS, axis=0)
    avg_PREDS = np.mean(PREDS, axis=0)

    data_output = pd.DataFrame({
        'weight_A': avg_WEIGHT_A,
        'weight_B': avg_WEIGHT_B,
        'bias': avg_BIAS,
        'pred_11': avg_PREDS[0, :],
        'pred_10': avg_PREDS[1, :],
        'pred_01': avg_PREDS[2, :],
        'pred_00': avg_PREDS[3, :],

    })

    data_output.to_csv(f'results/classic_cond_ratio-{ratio}.tsv', sep='\t', index=False)


def plot_ratio():
    data_01 = pd.read_csv('results/classic_cond_ratio-1.0.tsv', delimiter='\t')
    data_05 = pd.read_csv('results/classic_cond_ratio-5.0.tsv', delimiter='\t')
    data_25 = pd.read_csv('results/classic_cond_ratio-25.0.tsv', delimiter='\t')

    # Weights
    plt.figure()
    plt.plot(data_01['weight_A'])
    plt.plot(data_01['weight_B'], linestyle='--')
    plt.plot(data_05['weight_A'])
    plt.plot(data_05['weight_B'], linestyle='--')
    plt.plot(data_25['weight_A'])
    plt.plot(data_25['weight_B'], linestyle='--')

    plt.legend(['weight A (1:1)', 'weight B (1:1)', 'weight A (1:5)', 'weight B (1:5)', 'weight A (1:25)',
                'weight B (1:25)'])
    plt.title('Stimuli weight development')
    plt.xlabel('Epoch')
    plt.ylabel('Weight value')

    # Biases
    plt.figure()
    plt.plot(data_01['bias'])
    plt.plot(data_05['bias'])
    plt.plot(data_25['bias'])
    plt.legend(['(1:1)', '(1:5)', '(1:25)'])
    plt.xlabel('Epoch')
    plt.ylabel('Bias value')
    plt.title('Bias (threshold) development')

    # Predictions [1, 1]
    plt.figure()
    plt.plot(data_01['pred_11'])
    plt.plot(data_05['pred_11'])
    plt.plot(data_25['pred_11'])
    plt.hlines(1, 0, 30, linestyle='--', color='tab:red')
    plt.grid()
    plt.legend(['(1:1)', '(1:5)', '(1:25)', 'correct answer'])
    plt.xlabel('Epoch')
    plt.ylabel('Predicted value')
    plt.title('Prediction development on input [1, 1]')

    # Predictions [1, 0]
    plt.figure()
    plt.plot(data_01['pred_10'])
    plt.plot(data_05['pred_10'])
    plt.plot(data_25['pred_10'])
    plt.hlines(1, 0, 30, linestyle='--', color='tab:red')
    plt.grid()
    plt.legend(['(1:1)', '(1:5)', '(1:25)', 'correct answer'])
    plt.xlabel('Epoch')
    plt.ylabel('Predicted value')
    plt.title('Prediction development on input [1, 0]')

    # Predictions [0, 1]
    plt.figure()
    plt.plot(data_01['pred_01'])
    plt.plot(data_05['pred_01'])
    plt.plot(data_25['pred_01'])
    plt.hlines(0, 0, 30, linestyle='--', color='tab:red')
    plt.grid()
    plt.legend(['(1:1)', '(1:5)', '(1:25)', 'correct answer'])
    plt.xlabel('Epoch')
    plt.ylabel('Predicted value')
    plt.title('Prediction development on input [0, 1]')

    # Predictions [0, 0]
    plt.figure()
    plt.plot(data_01['pred_00'])
    plt.plot(data_05['pred_00'])
    plt.plot(data_25['pred_00'])
    plt.hlines(0, 0, 30, linestyle='--', color='tab:red')
    plt.grid()
    plt.legend(['(1:1)', '(1:5)', '(1:25)', 'correct answer'])
    plt.xlabel('Epoch')
    plt.ylabel('Predicted value')
    plt.title('Prediction development on input [0, 0]')

    plt.show()


# region <RUN>

def run():
    for ratio in [1.0, 5.0, 25.0]:
        run_experiment(ratio)


def plot_results():
    plot_ratio()


# endregion

if __name__ == '__main__':
    plot_results()
