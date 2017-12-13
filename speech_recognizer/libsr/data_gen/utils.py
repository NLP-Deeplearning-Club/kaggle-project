import json
import numpy as np
from .label_transform import label_transform
from .find_record import find_train_data

TRAIN_DATA, VALIDATION_DATA, TEST_DATA = find_train_data()

np.random.shuffle(TRAIN_DATA)


def _data_gen(perprocess, data, index_path, save=True, can_stop=False):
    """将数据在迭代器中进行预处理从而减小内存消耗,第一个next会返回数据集的长度.
    如果can_stop标记为False,数据一遍推完后它会回到起点再接着推

    Parameters:
        perprocess (callable): - 用于预处理的可执行对象
        data (Sequence): - 要预处理的原始数据
        index_path (file path): - 用于保存标签顺序的json文件地址
        save (bool): - 标识是保存标签顺序还是读取标签顺序
        can_stop (bool): - 标识是否会停止

    Yield:
        tuple[np.ndarray,np.ndarray]: - 由特征(n维)和标签onehot(一维)组成的元组,\
        特征维数要看预处理是怎么做的
    """
    ylen = len(data)
    yield ylen  # 返回数据集的长度
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
            try:
                y_yields = y_yields[label_index]
            except KeyError as ke:
                y_yields['silence'] = 0
                y_yields = y_yields[label_index]

    y_yields = y_yields.values
    y_yields = np.array(y_yields)
    flag = 0
    while True:
        if flag >= ylen:
            flag = 0
            if can_stop:
                raise StopIteration("iter stop")
            else:
                continue
        X_yield = perprocess(X[flag])
        y_yield = y_yields[flag]
        yield X_yield, y_yield
        flag += 1


def get_train_data(perprocess, index_path):
    """将训练集数据在迭代器中进行预处理后整合为一个(特征集,标签集)组成的训练数据集

    Parameters:
        perprocess (callable): - 用于预处理的可执行对象
        index_path (file path): - 用于保存标签顺序的json文件地址

    Return:
        tuple[np.ndarray,np.ndarray]: - 由特征(n维)组和标签onehot(一维)组组成的元组,\
        特征维数要看预处理是怎么做的
    """
    train_X = []
    train_y = []
    gen = _data_gen(perprocess, TRAIN_DATA, index_path, can_stop=True)
    next(gen)  # 让迭代器从第二个next开始执行
    for i, j in gen:
        train_X.append(i)
        train_y.append(j)
    X = np.array(train_X)
    y = np.array(train_y)
    return X, y


def get_validation_data(perprocess, index_path):
    """将验证集数据在迭代器中进行预处理后整合为一个(特征集,标签集)组成的验证数据集

    Parameters:
        perprocess (callable): - 用于预处理的可执行对象
        index_path (file path): - 用于保存标签顺序的json文件地址

    Return:
        tuple[np.ndarray,np.ndarray]: - 由特征(n维)组和标签onehot(一维)组组成的元组,\
        特征维数要看预处理是怎么做的
    """
    train_X = []
    train_y = []
    gen = _data_gen(perprocess, VALIDATION_DATA,
                    index_path, save=False, can_stop=True)
    next(gen)  # 让迭代器从第二个next开始执行
    for i, j in gen:
        train_X.append(i)
        train_y.append(j)
    X = np.array(train_X)
    y = np.array(train_y)
    return X, y


def get_test_data(perprocess, index_path):
    """将测试集数据在迭代器中进行预处理后整合为一个(特征集,标签集)组成的验证数据集

    Parameters:
        perprocess (callable): - 用于预处理的可执行对象
        index_path (file path): - 用于保存标签顺序的json文件地址

    Return:
        tuple[np.ndarray,np.ndarray]: - 由特征(n维)组和标签onehot(一维)组组成的元组,\
        特征维数要看预处理是怎么做的
    """
    train_X = []
    train_y = []
    gen = _data_gen(perprocess, TEST_DATA,
                    index_path, save=False, can_stop=True)
    next(gen)  # 让迭代器从第二个next开始执行
    for i, j in gen:
        train_X.append(i)
        train_y.append(j)
    X = np.array(train_X)
    y = np.array(train_y)
    return X, y


def train_gen(perprocess, batch_size, index_path):
    """将训练集数据在迭代器中进行预处理后按batch大小向模型中推送数据的生成器,
    第一个next会返回按batch划分后要多少个迭代才可以推完全部数据.这个迭代器是不会停止可以无限循环的,
    数据一遍推完后它会回到起点再接着推

    Parameters:
        perprocess (callable): - 用于预处理的可执行对象
        batch_size (int): - 设定最大batch长度
        index_path (file path): - 用于保存标签顺序的json文件地址

    Yield:
        tuple[np.ndarray,np.ndarray]: - 由特征组(batch_size*n维)\
        和标签onehot(batch_size*一维)组组成的元组,特征维数要看预处理是怎么做的
    """
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


def validation_gen(perprocess, batch_size, index_path):
    """将验证集数据在迭代器中进行预处理后按batch大小向模型中推送数据的生成器,
    第一个next会返回按batch划分后要多少个迭代才可以推完全部数据.这个迭代器是不会停止可以无限循环的,
    数据一遍推完后它会回到起点再接着推

    Parameters:
        perprocess (callable): - 用于预处理的可执行对象
        batch_size (int): - 设定最大batch长度
        index_path (file path): - 用于保存标签顺序的json文件地址

    Yield:
        tuple[np.ndarray,np.ndarray]: - 由特征组(batch_size*n维)\
        和标签onehot(batch_size*一维)组组成的元组,特征维数要看预处理是怎么做的
    """
    gen = _data_gen(perprocess, VALIDATION_DATA, index_path, save=False)
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
    """将验证集数据在迭代器中进行预处理后按batch大小向模型中推送数据的生成器,
    第一个next会返回按batch划分后要多少个迭代才可以推完全部数据.这个迭代器是不会停止可以无限循环的,
    数据一遍推完后它会回到起点再接着推

    Parameters:
        perprocess (callable): - 用于预处理的可执行对象
        batch_size (int): - 设定最大batch长度
        index_path (file path): - 用于保存标签顺序的json文件地址

    Yield:
        tuple[np.ndarray,np.ndarray]: - 由特征组(batch_size*n维)和\
        标签onehot(batch_size*一维)组组成的元组,特征维数要看预处理是怎么做的
    """
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
