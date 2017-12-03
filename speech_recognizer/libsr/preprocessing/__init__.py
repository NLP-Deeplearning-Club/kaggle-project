import numpy as np
import keras

from .resample import resample_from_path
#from .denoising import get_main_voice
from .vad import remove_muted
from .log_specgram import log_specgram

TEST_DATA = dict(x_train=np.random.random((1000, 20)),
                 y_train=keras.utils.to_categorical(
    np.random.randint(10, size=(1000, 1)), num_classes=10),
    x_test=np.random.random((100, 20)),
    y_test=keras.utils.to_categorical(
        np.random.randint(10, size=(100, 1)), num_classes=10)
)


def test_train_data_generator(batch_size):
    ylen = len(TEST_DATA.get("y_train"))
    loopcount = ylen // batch_size
    while True:
        i = np.random.randint(0, loopcount)
        yield TEST_DATA.get("x_train")[i * batch_size:(i + 1) * batch_size], TEST_DATA.get("y_train")[i * batch_size:(i + 1) * batch_size]


def test_test_data_generator(batch_size):

    ylen = len(TEST_DATA.get("y_test"))
    loopcount = ylen // batch_size
    while True:
        i = np.random.randint(0, loopcount)
        yield TEST_DATA.get("x_test")[i * batch_size:(i + 1) * batch_size], TEST_DATA.get("y_test")[i * batch_size:(i + 1) * batch_size]



def preprocessing_from_path(record_path, *, new_sample_rate=8000):
    """TODO
    """
    pass
