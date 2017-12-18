from pathlib import Path
from functools import partial
from keras import initializers
from speech_recognizer.libsr.preprocessing import (
    simple_mfcc_perprocess,
    mfcc_perprocess
)
from speech_recognizer.libsr.data_augmentation import (
    aug_process
)
from speech_recognizer.libsr.data_gen import TrainData
from speech_recognizer.libsr.models import build_cnn2d_model
from speech_recognizer.libsr.train import train_generator
from .utils import regist, get_current_function_name, tb_callback

per = partial(mfcc_perprocess, numcep=26, cnn=True)
aug = None

# per = partial(simple_mfcc_perprocess, numcep=26, cnn=False)
# aug = partial(aug_process,
#               desired_samples=16000,
#               time_shift=2000,
#               background_volume_range=0.1,
#               background_frequency=0.1)

# per = partial(mfcc_perprocess, numcep=26, cnn=False)
# aug = partial(aug_process,
#   desired_samples=16000,
#   time_shift=2000,
#   background_volume_range=0.1,
#   background_frequency=0.1)


@regist(per)
def cnn2d_process(data,
                  model_kwargs=dict(cnn_layer1={'input_shape': (99, 26, 1),
                                                "kernel_initializer": initializers.TruncatedNormal(
                      stddev=0.01),
                      "filters": 8,
                      "strides": (1, 1),
                      "kernel_size": 20,
                      'padding': 'same',
                      "activation": "relu"},
                      pool_layer1={"pool_size": (2, 2),
                                   "strides": (2, 2)},
                      dropout_layer1={"rate": 0.2},
                      cnn_layer2={"filters": 4,
                                  "kernel_initializer": initializers.TruncatedNormal(
        stddev=0.01),
        "kernel_size": 10,
        "strides": (1, 1),
        'padding': 'same',
        "activation": "relu"},
                      pool_layer2={"pool_size": (2, 2),
                                   "strides": (2, 2)},
                      dropout_layer2={"rate": 0.2},
                      mlp_layer1={"units": 12,
                                  "kernel_initializer": initializers.TruncatedNormal(
        stddev=0.01),
        "activation": "relu"}),

        optimizer='adam', loss='categorical_crossentropy',  # 训练用的参数
        metrics=['mae', 'accuracy'],
        train_batch_size=140,
        validation_batch_size=60, epochs=6):

    p = Path(__file__).absolute()
    _dir = p.parent.parent
    func_name = get_current_function_name()
    index_path = _dir.joinpath(
        "serialized_models/" + func_name + "_index.json")

    path = _dir.joinpath(
        "serialized_models/" + func_name + "_model.h5")

    data = TrainData(perprocess=per, index_path=index_path, aug_process=aug)
    # print(data.test_data[0].shape)
    # print(data.test_data[1].shape)

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
                                    callbacks=[tb_callback(func_name)]
                                    )
    trained_model.save(str(path))
    print("model save done!")
