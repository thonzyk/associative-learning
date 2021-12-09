from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import min_max_norm


def get_binary_perceptron(input_size, learning_rate=None, use_bias=True):
    optimizer = Adam(learning_rate=learning_rate) if learning_rate else Adam()

    model = Sequential([Dense(1,
                              activation='sigmoid',
                              use_bias=use_bias,
                              input_shape=(input_size,),
                              kernel_constraint=min_max_norm(-0., 10.),
                              bias_constraint=min_max_norm(-10., 10.)
                              )])

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
    )

    # model.summary()
    return model
