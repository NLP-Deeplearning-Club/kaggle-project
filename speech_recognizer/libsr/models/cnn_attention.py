from keras import activations
from keras.models import Model
from keras.layers import (
    Input,
    Convolution2D,
    Dense,
    Flatten,
    Dropout,
    MaxPooling2D,
    BatchNormalization,
    merge
)


def build_model(input_shape=(99, 81, 1)):
    inputs = Input(shape=input_shape)

    nor = BatchNormalization(input_shape=input_shape)(inputs)
    c1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(nor)
    c1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    dp1 = Dropout(0.2)(p1)

    c2 = Convolution2D(16, kernel_size=3, activation=activations.relu)(dp1)
    c2 = Convolution2D(16, kernel_size=3, activation=activations.relu)(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    dp2 = Dropout(0.2)(p2)

    c3 = Convolution2D(32, kernel_size=3, activation=activations.relu)(dp2)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    dp3 = Dropout(0.2)(p3)
    f1 = Flatten()(dp3)
    # ATTENTION PART STARTS HERE
    attention_probs = Dense(2240, activation='softmax',
                            name='attention_vec')(f1)
    attention_mul = merge([f1, attention_probs],
                          output_shape=2240, name='attention_mul', mode='mul')
    # ATTENTION PART FINISHES HERE
    attention_mul = Dense(128)(attention_mul)

    d1 = Dense(128, activation=activations.relu)(attention_mul)
    nor2 = BatchNormalization()(d1)
    output = Dense(31, activation='softmax')(nor2)
    model = Model(inputs=[inputs], outputs=output)
    return model