from functools import partial
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
from speech_recognizer.libsr.models import (
    build_lstm_my_self_dot_attention_model
)
from speech_recognizer.libsr.train import train_generator
from speech_recognizer.utils import vector_to_lab
from .utils import regist, tb_callback


per = normalize_perprocess
fe = partial(mfcc, numcep=26, cnn=False)


@regist(per, fe)
def lstm_my_self_attention_process(
        model_kwargs=dict(
            input_shape=(99, 26),
            lstm_layer={
                'units': 99,
                'return_sequences': True},
            self_attention_1d_layer={
                'similarity': "linear"}
        ),
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
                     repeat=5)
    train_gen = data.train_gen(train_batch_size)
    lenght = next(train_gen)
    validation_gen = data.validation_gen(validation_batch_size)
    steps = next(validation_gen)

    trained_model = train_generator(
        build_lstm_my_self_dot_attention_model(**model_kwargs),
        train_gen,
        steps_per_epoch=lenght,
        epochs=epochs,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        validation_data=validation_gen,
        validation_steps=steps,
        callbacks=[tb_callback(
            "lstm_my_self_attention_process")]
    )
    test_datas, test_label_vectors = data.test_data
    pre_lab = [vector_to_lab(i) for i in trained_model.predict(test_datas)]
    lab = [vector_to_lab(i) for i in test_label_vectors]
    z = zip(pre_lab, lab)
    for i in z:
        print(i)
    return trained_model
