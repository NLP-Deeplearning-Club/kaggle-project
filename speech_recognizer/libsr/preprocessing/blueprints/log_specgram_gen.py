"""先进行重采样,再将结果用于计算对数频谱,最后padding后返回频谱数据
"""
from ..find_record import find_train_data
from ..resample import resample_from_path
from ..padding import
import numpy as np

ALL_TRAIN_DATA = find_train_data()

np.random.shuffle(ALL_TRAIN_DATA)


def log_spec_gen(batch_size=50):
    ylen = len(ALL_TRAIN_DATA)
    loopcount = ylen // batch_size
    X = []
    y = []
    for f, t in ALL_TRAIN_DATA:
        X.append(f)
        y.append(t)

    while True:
        i = np.random.randint(0, loopcount)
        _X_yield = []
        for i in X[i * batch_size:(i + 1) * batch_size]:
            new_sample_rate, resampled = resample_from_path(i)
            _, _, spec = log_specgram(resampled, new_sample_rate)
            spec = padding(spec)
            _X_yield.append(spec)
        X_yield = np.array(_X_yield)
        y_yield = y[i * batch_size:(i + 1) * batch_size]
        yield X_yield, y_yield
