from keras.layers import (
    Input, Flatten,
    Permute, Reshape,
    Lambda, RepeatVector,
    merge, Dense,
    LSTM, Bidirectional
)
from keras.models import Model
from keras import backend as K
from attention_block import SelfAttention1DLayer


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
# def attention_3d_block(inputs, SINGLE_ATTENTION_VECTOR=False):
#     # inputs.shape = (batch_size, time_steps, input_dim)
#     input_dim = int(inputs.shape[2])
#     a = Permute((2, 1))(inputs)
#     a = Reshape((input_dim, TIME_STEPS))(a)
#     a = Dense(TIME_STEPS, activation='softmax')(a)
#     if SINGLE_ATTENTION_VECTOR:
#         a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
#         a = RepeatVector(input_dim)(a)
#     a_probs = Permute((2, 1), name='attention_vec')(a)
#     output_attention_mul = merge(
#         [inputs, a_probs], name='attention_mul', mode='mul')
#     return output_attention_mul


def build_model(input_shape=(99, 26),
                lstm_layer={
                    'units': 99,
                    'return_sequences': True},
                self_attention_1d_layer={
                    'kernel_size': (15, 30),
                    'similarity': "additive"}):
    inputs = Input(shape=input_shape)
    # Attention Layer
    att_out = SelfAttention1DLayer(**self_attention_1d_layer)(inputs)
    # RNN Layer
    rnn_out = Bidirectional(LSTM(**lstm_layer))(att_out)

    attention_mul = Flatten()(rnn_out)
    output = Dense(12, activation='softmax')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model
