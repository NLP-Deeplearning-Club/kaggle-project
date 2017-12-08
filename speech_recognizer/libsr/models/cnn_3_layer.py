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

def build_model(input_shape = (99, 81, 1)):
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Convolution2D(8, kernel_size=2, activation=activations.relu))
    model.add(Convolution2D(8, kernel_size=2, activation=activations.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(16, kernel_size=3, activation=activations.relu))
    model.add(Convolution2D(16, kernel_size=3, activation=activations.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(32, kernel_size=3, activation=activations.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(128, activation=activations.relu))
    model.add(BatchNormalization())
    model.add(Dense(128, activation=activations.relu))
    model.add(BatchNormalization())
    model.add(Dense(31, activation='softmax'))
    return model
