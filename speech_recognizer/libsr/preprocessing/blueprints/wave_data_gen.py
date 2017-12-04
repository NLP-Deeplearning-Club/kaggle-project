"""先进行重采样,再将结果返回
"""
from ..find_record import find_train_data
from ..resample import resample_from_path
import numpy as np

ALL_TRAIN_DATA = find_train_data()

np.random.shuffle(ALL_TRAIN_DATA)


def wave_gen(batch_size=50):
    ylen = len(ALL_TRAIN_DATA)
    loopcount = ylen // batch_size
    X = []
    y = []
    for f, t in ALL_TRAIN_DATA:
        X.append(f)
        y.append(t)
    while True:
        i = np.random.randint(0, loopcount)
        X_yield = np.array([resample_from_path(i)[1]
                            for i in X[i * batch_size:(i + 1) * batch_size]])
        y_yield = y[i * batch_size:(i + 1) * batch_size]
        yield X_yield, y_yield
