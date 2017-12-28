from keras.layers import (
    Input, Flatten,
    Permute, Reshape,
    Lambda, RepeatVector,
    merge, Dense, Dropout,
    LSTM, Bidirectional
)
from keras.models import Model
from keras import backend as K


def Attention3DLayer(time_step, single_attention_vector=False):
    def wrap(inputs):
        input_dim = int(inputs.shape[2])
        a = Permute((2, 1))(inputs)
        a = Reshape((input_dim, time_step))(a)
        a = Dense(time_step, activation='softmax')(a)
        if single_attention_vector:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        output_attention_mul = merge(
            [inputs, a_probs], name='attention_mul', mode='mul')
        return output_attention_mul
    return wrap


def build_model(input_shape=(99, 26),
                lstm_layer={
                    'units': 100,
                    'return_sequences': True},
                attention_3d_layer={
                    "time_step": 99,
                    "single_attention_vector": False}):

    inputs = Input(shape=input_shape)
    # RNN Layer
    rnn_out = Bidirectional(LSTM(**lstm_layer))(inputs)
    rnn_out = Dropout(0.2)(rnn_out)
    # Attention Layer
    attention_mul = Attention3DLayer(**attention_3d_layer)(rnn_out)
    attention_mul = Flatten()(attention_mul)
    attention_mul = Dropout(0.2)(attention_mul)
    output = Dense(12, activation='softmax')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model
