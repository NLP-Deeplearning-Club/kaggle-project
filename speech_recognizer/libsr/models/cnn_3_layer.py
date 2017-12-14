from keras import activations
from keras.models import Sequential
from keras.layers import (
    Convolution2D,
    Dense,
    Flatten,
    Dropout,
    MaxPooling2D,
    BatchNormalization
)


def build_model(input_shape=(99, 81, 1),
                cnn_layer1={"filters": 8, "kernel_size": 2,
                            "activation": "relu"},
                pool_layer1={"pool_size": (2, 2)},
                dropout_layer1={"rate": 0.2},
                cnn_layer2={"filters": 16, "kernel_size": 3,
                            "activation": "relu"},
                pool_layer2={"pool_size": (2, 2)},
                dropout_layer2={"rate": 0.2},
                cnn_layer3={"filters": 32, "kernel_size": 3,
                            "activation": "relu"},
                pool_layer3={"pool_size": (2, 2)},
                dropout_layer3={"rate": 0.2},
                mlp_layer1={"units": 128,
                            "activation": "relu"}):
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Convolution2D(**cnn_layer1))
    model.add(Convolution2D(**cnn_layer1))
    model.add(MaxPooling2D(**pool_layer1))
    model.add(Dropout(**dropout_layer1))

    model.add(Convolution2D(**cnn_layer2))
    model.add(Convolution2D(**cnn_layer2))
    model.add(MaxPooling2D(**pool_layer2))
    model.add(Dropout(**dropout_layer2))

    model.add(Convolution2D(**cnn_layer3))
    model.add(MaxPooling2D(**pool_layer3))
    model.add(Dropout(**dropout_layer3))
    model.add(Flatten())

    model.add(Dense(**mlp_layer1))
    model.add(BatchNormalization())
    model.add(Dense(**mlp_layer1))
    model.add(BatchNormalization())
    model.add(Dense(12, activation='softmax'))
    return model
