"""先进行重采样,再将结果用于计算对数频谱,最后padding后返回频谱数据
"""
import json
import numpy as np
from scipy.io import wavfile
from .utils import regist
from ..log_specgram import log_specgram
from ..find_record import find_train_data
from ..resample import resample
from ..padding import padding_wave
from ..label_transform import label_transform

TRAIN_DATA, TEST_DATA = find_train_data()

np.random.shuffle(TRAIN_DATA)

@regist
def log_spec_perprocess(path):
    sample_rate, samples = wavfile.read(path)
    samples = padding_wave(samples)
    if len(samples) > 16000:
        samples = samples[:16000]
    else:
        samples = samples
    new_sample_rate, resampled = resample(samples, sample_rate)
    _, _, specgram = log_specgram(
        resampled, sample_rate=new_sample_rate)
    X_yield = specgram
    X_yield = X_yield.reshape(tuple(list(X_yield.shape) + [1]))
    return X_yield


def _data_gen(data, index_path=None):
    ylen = len(data)
    yield ylen
    X = []
    y = []
    for f, t in data:
        X.append(f)
        y.append(t)
    y_yields = label_transform(y)
    label_index = list(y_yields.columns.values)
    if index_path:
        with open(str(index_path), "w") as f:
            json.dump(label_index, f)
    y_yields = y_yields.values
    y_yields = np.array(y_yields)
    flag = 0
    while True:
        if flag >= ylen:
            flag = 0
            continue
        try:
            X_yield = log_spec_perprocess(X[flag])
        except:
            continue
        y_yield = y_yields[flag]
        yield X_yield, y_yield
        flag += 1


def get_train_data(index_path):
    train_X = []
    train_y = []
    gen = _data_gen(TRAIN_DATA, index_path)
    l = next(gen)
    for i, j in gen:
        train_X.append(i)
        train_y.append(j)
    X = np.array(train_X)
    y = np.array(train_y)
    return X, y


def log_spec_train_gen(batch_size, index_path):
    gen = _data_gen(TRAIN_DATA, index_path)
    ylen = next(gen)
    loopcount = ylen // batch_size
    yield loopcount
    while True:
        batch_X = []
        batch_y = []
        for i in range(batch_size):
            temp_X, temp_y = next(gen)
            batch_X.append(temp_X)
            batch_y.append(temp_y)
        yield np.array(batch_X), np.array(batch_y)


def log_spec_test_gen(batch_size, index_path=None):
    gen = _data_gen(TEST_DATA, index_path)
    ylen = next(gen)
    loopcount = ylen // batch_size
    yield loopcount
    while True:
        batch_X = []
        batch_y = []
        for i in range(batch_size):
            temp_X, temp_y = next(gen)
            batch_X.append(temp_X)
            batch_y.append(temp_y)
        yield np.array(batch_X), np.array(batch_y)
