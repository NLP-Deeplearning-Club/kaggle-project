from functools import partial
from keras import initializers
from speech_recognizer.libsr.preprocessing import (
    normalize_perprocess
)
from speech_recognizer.libsr.data_augmentation import (
    aug_process
)
from speech_recognizer.libsr.feature_extract import (
    mfcc
)
from speech_recognizer.libsr.data_gen import TrainData
from speech_recognizer.libsr.models import build_cnn2d_model
from speech_recognizer.libsr.train import train_generator
from .utils import regist, tb_callback

per = normalize_perprocess
fe = partial(mfcc, numcep=26, cnn=True)


@regist(per, fe)
def cnn2d_process(
        model_kwargs=dict(
            cnn_layer1={
                'input_shape': (99, 26, 1),
                "kernel_initializer": initializers.TruncatedNormal(
                    stddev=0.01),
                "filters": 8,
                "strides": (1, 1),
                "kernel_size": 20,
                'padding': 'same',
                "activation": "relu"},
            pool_layer1={
                "pool_size": (2, 2),
                "strides": (2, 2)},
            dropout_layer1={"rate": 0.2},
            cnn_layer2={
                "filters": 4,
                "kernel_initializer": initializers.TruncatedNormal(
                    stddev=0.01),
                "kernel_size": 10,
                "strides": (1, 1),
                'padding': 'same',
                "activation": "relu"},
            pool_layer2={
                "pool_size": (2, 2),
                "strides": (2, 2)},
            dropout_layer2={"rate": 0.2},
            mlp_layer1={
                "units": 12,
                "kernel_initializer": initializers.TruncatedNormal(
                    stddev=0.01),
                "activation": "relu"}
        ),
        aug_process_kwargs=dict(
            time_shift=2000,
            background_volume_range=0.1,
            background_frequency=0.1),  # 数据增强
        optimizer='adam', loss='categorical_crossentropy',  # 训练用的参数
        metrics=['mae', 'accuracy'],  # 观测的参数
        train_batch_size=140,  # 训练的batchsize
        validation_batch_size=60,  # 验证集的batchsize
        epochs=6):

    if aug_process_kwargs:
        aug = partial(aug_process, **aug_process_kwargs)
    else:
        aug = None
    data = TrainData(perprocess=per, feature_extract=fe, aug_process=aug)
    train_gen = data.train_gen(train_batch_size)
    lenght = next(train_gen)
    validation_gen = data.validation_gen(validation_batch_size)
    steps = next(validation_gen)
    trained_model = train_generator(build_cnn2d_model(**model_kwargs),
                                    train_gen,
                                    steps_per_epoch=lenght,
                                    epochs=epochs,
                                    optimizer=optimizer,
                                    loss=loss,
                                    metrics=metrics,
                                    validation_data=validation_gen,
                                    validation_steps=steps,
                                    callbacks=[tb_callback("cnn2d_process")]
                                    )
    return trained_model
