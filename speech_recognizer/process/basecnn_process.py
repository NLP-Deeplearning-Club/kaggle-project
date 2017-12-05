"""示范用的测试训练过程
"""
from pathlib import Path
from keras.optimizers import Adam
from speech_recognizer.libsr.preprocessing.blueprints.log_specgram_gen import (
    log_spec_train_gen,
    log_spec_test_gen,
    get_train_data
)
from speech_recognizer.libsr.models import basecnn_model
from speech_recognizer.libsr.train import train, train_generator
from .utils import regist


@regist
def basecnn_process():
    X, y = get_train_data()
    print(X.shape)
    print(y.shape)
    trained_model = train(basecnn_model, X, y, 1, 16,
                          optimizer=Adam(),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    test_gen = log_spec_test_gen(16)
    steps = next(test_gen)
    print("evaluate score:")
    print(trained_model.evaluate_generator(
        test_gen, steps=steps))

    p = Path(__file__).absolute()
    _dir = p.parent.parent
    path = _dir.joinpath("serialized_models/basecnn_process_model.h5")
    trained_model.save(str(path))


@regist
def basecnn_gen_process():
    train_gen = log_spec_train_gen(16)
    lenght = next(train_gen)
    print(lenght)
    trained_model = train_generator(basecnn_model, train_gen, steps_per_epoch=lenght, epochs=1,
                                    optimizer=Adam(),
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'])
    test_gen = log_spec_test_gen(16)
    steps = next(test_gen)
    print("evaluate score:")
    print(trained_model.evaluate_generator(
        test_gen, steps=steps))

    p = Path(__file__).absolute()
    _dir = p.parent.parent
    path = _dir.joinpath("serialized_models/basecnn_gen_process_model.h5")
    trained_model.save(str(path))
