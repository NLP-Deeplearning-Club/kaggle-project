from functools import partial
from keras import regularizers
from keras.callbacks import EarlyStopping
from speech_recognizer.libsr.preprocessing import (
    normalize_perprocess
)
from speech_recognizer.libsr.data_augmentation import(
    aug_process
)
from speech_recognizer.libsr.feature_extract import (
    mfcc
)
from speech_recognizer.libsr.data_gen import TrainData
from speech_recognizer.libsr.models import build_lstm_attention_model
from speech_recognizer.libsr.train import train_generator
from speech_recognizer.utils import vector_to_lab
from .utils import regist, tb_callback


per = normalize_perprocess
fe = partial(mfcc, numcep=26, cnn=False)


@regist(per, fe)
def lstm_attention_process(
        model_kwargs=dict(
            input_shape=(99, 26),
            lstm_layer={
                'units': 100,
                'return_sequences': True,
                'kernel_regularizer': regularizers.l2(0.01),
            },
            attention_3d_layer={
                "time_step": 99,
                "single_attention_vector": False}),
        aug_process_kwargs=dict(
            time_shift=2000,
            background_volume_range=0.1,
            background_frequency=0.1),  # 数据增强
        optimizer='adam', loss='categorical_crossentropy',  # 训练用的参数
        metrics=['mae', 'accuracy'], train_batch_size=140,
        validation_batch_size=60, epochs=4):
    if aug_process_kwargs:
        aug = partial(aug_process, **aug_process_kwargs)
    else:
        aug = None
    data = TrainData(perprocess=per,
                     feature_extract=fe,
                     aug_process=aug,
                     validation_rate=0.2,
                     test_rate=0.01,
                     repeat=5)
    train_gen = data.train_gen(train_batch_size)
    lenght = next(train_gen)
    validation_gen = data.validation_gen(validation_batch_size)
    steps = next(validation_gen)

    trained_model = train_generator(build_lstm_attention_model(**model_kwargs),
                                    train_gen,
                                    steps_per_epoch=lenght,
                                    epochs=epochs,
                                    optimizer=optimizer,
                                    loss=loss,
                                    metrics=metrics,
                                    validation_data=validation_gen,
                                    validation_steps=steps,
                                    callbacks=[
                                        tb_callback(
                                            "lstm_attention_process"),
                                        EarlyStopping(
                                            monitor='val_loss',
                                            patience=0,
                                            verbose=0,
                                            mode='auto')]
                                    )
    test_datas, test_label_vectors = data.test_data
    pre_lab = [vector_to_lab(i) for i in trained_model.predict(test_datas)]
    lab = [vector_to_lab(i) for i in test_label_vectors]
    z = zip(pre_lab, lab)
    for i in z:
        print(i)
    return trained_model
