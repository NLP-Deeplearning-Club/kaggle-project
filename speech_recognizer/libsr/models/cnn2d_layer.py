# from keras import activations
from keras import initializers
from keras.models import Sequential
from keras.layers import (
    Convolution2D,
    Dense,
    Flatten,
    Dropout,
    MaxPooling2D
)


def build_model(
    cnn_layer1={'input_shape': (99, 81, 1),
                "kernel_initializer": initializers.TruncatedNormal(
                    stddev=0.01),
                "filters": 8,
                "strides": (1, 1),
                "kernel_size": 20,
                'padding': 'same',
                "activation": "relu"},
    pool_layer1={"pool_size": (2, 2),
                 "strides": (2, 2)},
    dropout_layer1={"rate": 0.2},
    cnn_layer2={"filters": 4,
                "kernel_initializer": initializers.TruncatedNormal(
        stddev=0.01),
        "kernel_size": 10,
        "strides": (1, 1),
        'padding': 'same',
        "activation": "relu"},
    pool_layer2={"pool_size": (2, 2),
                 "strides": (2, 2)},
    dropout_layer2={"rate": 0.2},
    mlp_layer1={"units": 12,
                "kernel_initializer": initializers.TruncatedNormal(
        stddev=0.01),
        "activation": "relu"}):
    model = Sequential()
    # model.add(BatchNormalization(input_shape=input_shape))
    # model.add(Input(input_shape=input_shape))
    model.add(Convolution2D(**cnn_layer1))
    model.add(Dropout(**dropout_layer1))
    model.add(MaxPooling2D(**pool_layer1))

    model.add(Convolution2D(**cnn_layer2))
    model.add(MaxPooling2D(**pool_layer2))
    model.add(Dropout(**dropout_layer2))

    model.add(Flatten())

    model.add(Dense(**mlp_layer1))
    model.add(Dense(12, activation='softmax'))
    return model
