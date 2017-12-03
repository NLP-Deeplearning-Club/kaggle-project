import numpy as np
import keras

from .resample import resample_from_path
from .denoising import get_main_voice
from .vad import remove_muted
from .log_specgram import log_specgram


def test_data_generator():
    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(
        np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(
        np.random.randint(10, size=(100, 1)), num_classes=10)
    return x_train, y_train, x_test, y_test


def preprocessing_from_path(record_path, *, new_sample_rate=8000):
    """TODO
    """
    pass
