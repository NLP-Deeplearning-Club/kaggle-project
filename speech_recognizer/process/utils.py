import inspect
from pathlib import Path
import keras
p = Path(__file__).absolute()
_dir = p.parent.parent
REGIST_PROCESS = {}
REGIST_PERPROCESS = {}

tb_cb = keras.callbacks.TensorBoard(
    log_dir=str(_dir.joinpath("tmp/log")), write_images=1, histogram_freq=1)


def get_current_function_name():
    return inspect.stack()[1][3]


class regist:
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, func):
        REGIST_PROCESS[func.__name__] = func
        REGIST_PERPROCESS[func.__name__] = self.preprocess
        return func
