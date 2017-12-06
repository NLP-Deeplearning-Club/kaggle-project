import json
import numpy as np
from ..label_transform import label_transform
from ..find_record import find_train_data

TRAIN_DATA, TEST_DATA = find_train_data()

np.random.shuffle(TRAIN_DATA)


def _data_gen(perprocess, data, index_path, save=True):
    ylen = len(data)
    yield ylen
    X = []
    y = []
    for f, t in data:
        X.append(f)
        y.append(t)
    y_yields = label_transform(y)
    if save:
        label_index = list(y_yields.columns.values)
        with open(str(index_path), "w") as f:
            json.dump(label_index, f)
    else:
        with open(str(index_path), "r") as f:
            label_index = json.load(f)
            y_yields = y_yields[label_index]
    y_yields = y_yields.values
    y_yields = np.array(y_yields)
    flag = 0
    while True:
        if flag >= ylen:
            flag = 0
            continue

        X_yield = perprocess(X[flag])

        y_yield = y_yields[flag]
        yield X_yield, y_yield
        flag += 1


def get_train_data(perprocess, index_path):
    train_X = []
    train_y = []
    gen = _data_gen(perprocess, TRAIN_DATA, index_path)
    next(gen)
    for i, j in gen:
        train_X.append(i)
        train_y.append(j)
    X = np.array(train_X)
    y = np.array(train_y)
    return X, y


def get_test_data(perprocess, index_path):
    train_X = []
    train_y = []
    gen = _data_gen(perprocess, TEST_DATA, index_path, save=False)
    next(gen)
    for i, j in gen:
        train_X.append(i)
        train_y.append(j)
    X = np.array(train_X)
    y = np.array(train_y)
    return X, y


def train_gen(perprocess, batch_size, index_path):
    gen = _data_gen(perprocess, TRAIN_DATA, index_path)
    ylen = next(gen)
    loopcount = ylen // batch_size
    yield loopcount
    while True:
        batch_X = []
        batch_y = []
        for _ in range(batch_size):
            temp_X, temp_y = next(gen)
            batch_X.append(temp_X)
            batch_y.append(temp_y)
        yield np.array(batch_X), np.array(batch_y)


def test_gen(perprocess, batch_size, index_path):
    gen = _data_gen(perprocess, TEST_DATA, index_path, save=False)
    ylen = next(gen)
    loopcount = ylen // batch_size
    yield loopcount
    while True:
        batch_X = []
        batch_y = []
        for _ in range(batch_size):
            temp_X, temp_y = next(gen)
            batch_X.append(temp_X)
            batch_y.append(temp_y)
        yield np.array(batch_X), np.array(batch_y)
