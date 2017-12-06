"""示范用的测试训练过程,注意,要使用预处理过程函数来初始化装饰器,这样才能在命令行中显示
"""
from pathlib import Path
from keras.optimizers import Adam
from speech_recognizer.libsr.preprocessing.blueprints import (
    log_spec_perprocess,
    log_spec_train_gen,
    log_spec_test_gen
)
from speech_recognizer.libsr.models import basecnn_model
from speech_recognizer.libsr.train import train_generator
from .utils import regist


@regist(log_spec_perprocess)
def cnn_spec_gen_process(batch_size=16, epochs=5):
    p = Path(__file__).absolute()
    _dir = p.parent.parent
    index_path = _dir.joinpath(
        "serialized_models/basecnn_gen_process_index.json")
    train_gen = log_spec_train_gen(batch_size, index_path)
    lenght = next(train_gen)
    print(lenght)
    trained_model = train_generator(basecnn_model,
                                    train_gen,
                                    steps_per_epoch=lenght,
                                    epochs=epochs,
                                    optimizer=Adam(),
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy']
                                    )
    test_gen = log_spec_test_gen(16, index_path)
    steps = next(test_gen)
    print("evaluate score:")
    print("steps:{steps}".format(steps=steps))
    print(trained_model.evaluate_generator(
        test_gen, steps=steps))

    path = _dir.joinpath("serialized_models/basecnn_gen_process_model.h5")
    trained_model.save(str(path))
