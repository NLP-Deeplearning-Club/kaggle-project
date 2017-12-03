from pathlib import Path
from speech_recognizer.libsr.preprocessing import test_train_data_generator, test_test_data_generator
from speech_recognizer.libsr.models import test_model
from speech_recognizer.libsr.train import train, train_generator
from .utils import regist


@regist
def test_process():
    trained_model = train_generator(test_model, test_train_data_generator(20),
                                    steps_per_epoch=300, epochs=1,
                                    loss='binary_crossentropy',
                                    optimizer='rmsprop',
                                    metrics=['accuracy'])

    print("evaluate score:")
    print(trained_model.evaluate_generator(test_test_data_generator(20), steps=10))

    p = Path(__file__).absolute()
    _dir = p.parent.parent
    path = _dir.joinpath("serialized_models/test_process_model.h5")
    trained_model.save(str(path))
