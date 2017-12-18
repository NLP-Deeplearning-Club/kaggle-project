import json
from pathlib import Path
from functools import partial
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
from .utils import regist, get_current_function_name, tb_callback


per = normalize_perprocess
fe = partial(mfcc, numcep=26, cnn=False)


@regist(per, fe)
def lstm_attention_process(
        model_kwargs=dict(input_shape=(99, 26),
                          lstm_layer={
            'units': 100,
            'return_sequences': True},
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

    p = Path(__file__).absolute()
    _dir = p.parent.parent
    func_name = get_current_function_name()
    index_path = _dir.joinpath(
        "serialized_models/" + func_name + "_index.json")

    path = _dir.joinpath(
        "serialized_models/" + func_name + "_model.h5")
    if aug_process_kwargs:
        aug = partial(aug_process, **aug_process_kwargs)
    else:
        aug = None
    data = TrainData(perprocess=per, feature_extract=fe,
                     index_path=index_path, aug_process=aug)
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
                                    callbacks=[tb_callback(func_name)]
                                    )
    trained_model.save(str(path))
    print("model save done!")
    test_data, test_label = data.test_data
    with open(str(index_path)) as f:
        labels = json.load(f)

    pre_lab = [max(zip(labels, i), key=lambda x:x[1])
               for i in trained_model.predict(test_data)][0]
    lab = [max(zip(labels, i), key=lambda x:x[1]) for i in test_label][0]
    z = zip(pre_lab, lab)
    for i in z:
        print(i)
