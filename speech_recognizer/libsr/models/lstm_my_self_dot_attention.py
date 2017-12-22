from keras.layers import (
    Input, Flatten, Dense,
    Dropout, LSTM, Bidirectional
)
from keras.models import Model
from keras_attention_block import SelfAttention1DLayer


def build_model(input_shape=(99, 26),
                lstm_layer={
                    'units': 99,
                    'return_sequences': True},
                self_attention_1d_layer={
                    'similarity': "dot_product"}):
    inputs = Input(shape=input_shape)
    # RNN Layer
    rnn_out = Bidirectional(LSTM(**lstm_layer))(inputs)
    # Dropout
    rnn_out = Dropout(0.2)(rnn_out)
    # Attention Layer
    att_out = SelfAttention1DLayer(**self_attention_1d_layer)(rnn_out)

    attention_mul = Flatten()(att_out)
    output = Dense(12, activation='softmax')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model
