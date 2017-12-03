from pathlib import Path
from speech_recognizer.libsr.preprocessing import test_data_generator
from speech_recognizer.libsr.models import test_model
from speech_recognizer.libsr.train import train
from .utils import regist

@regist
def test_process():
    x_train, y_train, x_test, y_test = test_data_generator()

    trained_model = train(model, x_train, y_train,
                        epochs=20,
                        batch_size=128,
                        loss='binary_crossentropy',
                        optimizer='rmsprop',
                        metrics=['accuracy'])

    print("evaluate score:")
    print(trained_model.evaluate(x_test, y_test))

    p = Path(__file__).absolute()
    _dir = p.parent
    path = _dir.joinpath("serialized_models/test_process_model.h5")
    trained_model.save(str(path))

test_process()