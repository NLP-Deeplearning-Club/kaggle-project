from pathlib import Path
from functools import partial
from speech_recognizer.libsr.preprocessing import (
    simple_mfcc_perprocess,
    mfcc_perprocess
)
from speech_recognizer.libsr.data_augmentation import(
    aug_process
)
from speech_recognizer.libsr.data_gen import TrainData
from speech_recognizer.libsr.models import build_lstm_attention_model
from speech_recognizer.libsr.train import train_generator
from .utils import regist, get_current_function_name, tb_callback

# per = partial(mfcc_perprocess, numcep=26, cnn=False)
# aug = None

# per = partial(simple_mfcc_perprocess, numcep=26, cnn=False)
# aug = partial(aug_process,
#               desired_samples=16000,
#               time_shift=2000,
#               background_volume_range=0.1,
#               background_frequency=0.1)

per = partial(mfcc_perprocess, numcep=26, cnn=False)
aug = partial(aug_process,
              desired_samples=16000,
              time_shift=2000,
              background_volume_range=0.1,
              background_frequency=0.1)


@regist(per)
def lstm_attention_process(
        model_kwargs=dict(input_shape=(99, 26),
                          lstm_layer={
            'units': 100,
            'return_sequences': True},
            attention_3d_layer={
                "time_step": 99,
            "single_attention_vector": False}),
        optimizer='adam', loss='categorical_crossentropy',  # 训练用的参数
        metrics=['mae', 'accuracy'], train_batch_size=140,
        validation_batch_size=60, epochs=6):

    p = Path(__file__).absolute()
    _dir = p.parent.parent
    func_name = get_current_function_name()
    index_path = _dir.joinpath(
        "serialized_models/" + func_name + "_index.json")

    path = _dir.joinpath(
        "serialized_models/" + func_name + "_model.h5")

    data = TrainData(perprocess=per, index_path=index_path, aug_process=aug)
    #print(data.test_data[0].shape)
    #print(data.test_data[1].shape)

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
