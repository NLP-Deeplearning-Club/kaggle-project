"""先进行重采样,再将结果用于计算对数频谱,最后padding后返回频谱数据
"""
import numpy as np
from scipy.io import wavfile
from ..log_specgram import log_specgram
from ..find_record import find_train_data
from ..resample import resample
from ..padding import padding_wave, chop_audio_gen
from ..label_transform import label_transform

TRAIN_DATA, TEST_DATA = find_train_data()

np.random.shuffle(TRAIN_DATA)


def _data_gen(data):
    ylen = len(data)
    yield ylen
    X = []
    y = []
    for f, t in data:
        X.append(f)
        y.append(t)

    y_yields = label_transform(y)
    label_index = y_yields.columns.values
    y_yields = y_yields.values
    y_yields = np.array(y_yields)
    datas = list(zip(X, y_yields))
    flag = 0
    while True:
        if flag >= ylen:
            flag = 0
            continue
        sample_rate, samples = wavfile.read(datas[flag][0])
        samples = padding_wave(samples)
        if len(samples) > 16000:
            n_samples = chop_audio_gen(samples)
        else:
            n_samples = [samples]
        for samples in n_samples:
            new_sample_rate, resampled = resample(samples, sample_rate)
            _, _, specgram = log_specgram(
                resampled, sample_rate=new_sample_rate)
            X_yield = specgram
            X_yield = X_yield.reshape(tuple(list(X_yield.shape) + [1]))
            yield X_yield, datas[flag][1]


def get_train_data():
    train_X = []
    train_y = []
    gen = _data_gen(TRAIN_DATA)
    l = next(gen)
    for i, j in gen:
        train_X.append(i)
        train_y.append(j)
    X = np.array(train_X)
    y = np.array(train_y)
    return X, y


def log_spec_train_gen(batch_size):
    gen = _data_gen(TRAIN_DATA)
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


def log_spec_test_gen(batch_size):
    gen = _data_gen(TEST_DATA)
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
