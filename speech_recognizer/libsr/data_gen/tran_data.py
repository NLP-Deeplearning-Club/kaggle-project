import json
import numpy as np
from scipy.io import wavfile
from .label_transform import label_transform
from .find_record import find_data


class TrainData:
    """用于生成用来训练的数据的类,perprocess和aug_process都要求是可执行对象,且参数只有固定值,\
    可以使用标准库中的偏函数(from functools import partial)或者函数默认参数实现,推荐第一种,这样便于调整参数

    Attributes:
        perprocess (callable): - 用于预处理的可执行对象,要求输入的参数为sample_rate, samples
        index_path (file path): - 用于保存标签顺序的json文件地址
        aug_process (callable): - 用于做数据增强的过程,默认为None,也就是不做数据增强,如果是可执行程序,\
        要求参数为wav, label, mode
        TRAIN_DATA (tuple): - 训练集原始数据,分为两部分,一部分是数据地址,一部分是标签
        VALIDATION_DATA (tuple): - 验证集原始数据,分为两部分,一部分是数据地址,一部分是标签
        TEST_DATA (tuple): - 测试集原始数据,分为两部分,一部分是数据地址,一部分是标签

    Property:
        train_data (tuple[np.ndarry,np.ndarry]): - 由特征矩阵,标签矩阵组成的训练集数据
        validation_data (tuple[np.ndarry,np.ndarry]): - 由特征矩阵,标签矩阵组成的验证集数据
        test_data (tuple[np.ndarry,np.ndarry]): - 由特征矩阵,标签矩阵组成的测试集数据
    """

    @staticmethod
    def _data_gen(perprocess, feature_extract, data,
                  index_path, aug_process=None, aug_mode=None,
                  save=True, can_stop=False):
        """将数据在迭代器中进行预处理从而减小内存消耗,第一个next会返回数据集的长度.
        如果can_stop标记为False,数据一遍推完后它会回到起点再接着推

        Parameters:
            perprocess (callable): - 用于预处理的可执行对象
            feature_extract (callable): - 用于提取特征的可执行对象
            aug_process (callable): - 用于数据增强的可执行对象
            aug_mode (str): - 用于指明数据增强的模式,默认为None,train模式会和其他有区别
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
                except KeyError:
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
            X_yield_path = X[flag]
            y_yield_label = y[flag]
            X_yield_rate, X_yield_wav = wavfile.read(X_yield_path)

            X_yield_rate, X_yield_wav = perprocess(
                sample_rate=X_yield_rate, samples=X_yield_wav)
            if aug_process:
                X_yield_wav = aug_process(
                    wav=X_yield_wav, label=y_yield_label, mode=aug_mode)
            X_yield = feature_extract(X_yield_rate, X_yield_wav)
            y_yield = y_yields[flag]
            yield X_yield, y_yield
            flag += 1

    def __init__(self, perprocess,
                 feature_extract,
                 index_path,
                 aug_process=None,
                 validation_rate=0.1,
                 test_rate=0.1,
                 repeat=0,
                 unknown_rate=0.1,
                 silence_rate=0.1):
        """
        Parameters:
            perprocess (callable): - 用于预处理的可执行对象
            feature_extract (callable): - 用于提取特征的可执行对象
            index_path (file path): - 用于保存标签顺序的json文件地址
            aug_process (callable): - 用于做数据增强的过程,默认为None,也就是不做数据增强
            validation_rate (float): - 指明验证集比例,这个并不是严格的,而是使用random函数随机抽取.默认为0.1
            test_rate (float): - 指明测试集比例,这个并不是严格的,而是使用random函数随机抽取.默认为0.1
            repeat (int): - 指明训练集中非unknown和非silence的数据重复多少次
            unknown_rate (float): - 指明各个数据集中未知数据的比例.默认为0.1
            silence_rate (float): - 指明各个数据集中沉默音的比例.默认为0.1
        """
        self.perprocess = perprocess
        self.feature_extract = feature_extract
        self.index_path = index_path
        self.aug_process = aug_process
        self.TRAIN_DATA, self.VALIDATION_DATA, self.TEST_DATA = find_data(
            validation_rate=validation_rate,
            test_rate=test_rate,
            repeat=repeat,
            unknown_rate=unknown_rate,
            silence_rate=silence_rate)
        self._train_data = None
        self._validation_data = None
        self._test_data = None

    @property
    def train_data(self):
        if self._train_data is None:
            self._train_data = self._get_data(
                self.TRAIN_DATA, aug_mode="train", save=True)
        return self._train_data

    @property
    def validation_data(self):
        if self._validation_data is None:
            self._validation_data = self._get_data(self.VALIDATION_DATA)
        return self._validation_data

    @property
    def test_data(self):
        if self._test_data is None:
            self._test_data = self._get_data(self.TEST_DATA)
        return self._test_data

    def _get_data(self, dataset, aug_mode=None, save=False):
        """将训练集数据在迭代器中进行预处理后整合为一个(特征集,标签集)组成的训练数据集

        Parameters:
            dataset (tuple): - 要处理的原始数据集
            aug_mode (str): - 用于指明数据增强的模式,默认为None,train模式会和其他有区别
            save (bool): - 是否保存标签顺序的json文件地址

        Return:
            tuple[np.ndarray,np.ndarray]: - 由特征(n维)组和标签onehot(一维)组组成的元组,\
            特征维数要看预处理是怎么做的
        """
        train_X = []
        train_y = []
        gen = TrainData._data_gen(self.perprocess,
                                  self.feature_extract,
                                  dataset,
                                  self.index_path,
                                  self.aug_process,
                                  aug_mode=aug_mode,
                                  save=save, can_stop=True)
        next(gen)  # 让迭代器从第二个next开始执行
        for i, j in gen:
            train_X.append(i)
            train_y.append(j)
        X = np.array(train_X)
        y = np.array(train_y)
        # print(X.shape)
        return X, y

    def _batch_gen(self, dataset, batch_size, aug_mode=None, save=False):
        """将数据集数据在迭代器中进行预处理后按batch大小向模型中推送数据的生成器,
        第一个next会返回按batch划分后要多少个迭代才可以推完全部数据.这个迭代器是不会停止可以无限循环的,
        数据一遍推完后它会回到起点再接着推

        Parameters:
            dataset (tuple[list,list]): - 由音频地址集,和标签集组成的数据集
            batch_size (int): - 设定最大batch长度
            aug_mode (str): - 用于指明数据增强的模式,默认为None,train模式会和其他有区别
            save (bool): - 是否保存标签顺序的json文件地址

        Yield:
            tuple[np.ndarray,np.ndarray]: - 由特征组(batch_size*n维)\
            和标签onehot(batch_size*一维)组组成的元组,特征维数要看预处理是怎么做的
        """
        gen = TrainData._data_gen(self.perprocess, self.feature_extract, dataset,
                                  self.index_path, self.aug_process,
                                  aug_mode=aug_mode, save=save)
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
            X = np.array(batch_X)
            y = np.array(batch_y)
            yield X, y

    def train_gen(self, batch_size):
        """将训练集数据在迭代器中进行预处理后按batch大小向模型中推送数据的生成器,
        第一个next会返回按batch划分后要多少个迭代才可以推完全部数据.这个迭代器是不会停止可以无限循环的,
        数据一遍推完后它会回到起点再接着推

        Parameters:
            batch_size (int): - 设定最大batch长度

        Yield:
            tuple[np.ndarray,np.ndarray]: - 由特征组(batch_size*n维)\
            和标签onehot(batch_size*一维)组组成的元组,特征维数要看预处理是怎么做的
        """
        batch_gen = self._batch_gen(
            self.TRAIN_DATA, batch_size, aug_mode="train", save=True)
        loopcount = next(batch_gen)
        yield loopcount
        while True:
            batch_X, batch_y = next(batch_gen)
            yield batch_X, batch_y

    def validation_gen(self, batch_size):
        """将验证集数据在迭代器中进行预处理后按batch大小向模型中推送数据的生成器,
        第一个next会返回按batch划分后要多少个迭代才可以推完全部数据.这个迭代器是不会停止可以无限循环的,
        数据一遍推完后它会回到起点再接着推

        Parameters:
            batch_size (int): - 设定最大batch长度

        Yield:
            tuple[np.ndarray,np.ndarray]: - 由特征组(batch_size*n维)\
            和标签onehot(batch_size*一维)组组成的元组,特征维数要看预处理是怎么做的
        """
        batch_gen = self._batch_gen(self.VALIDATION_DATA, batch_size)
        loopcount = next(batch_gen)
        yield loopcount
        while True:
            batch_X, batch_y = next(batch_gen)
            yield batch_X, batch_y

    def test_gen(self, batch_size):
        """将测试集数据在迭代器中进行预处理后按batch大小向模型中推送数据的生成器,
        第一个next会返回按batch划分后要多少个迭代才可以推完全部数据.这个迭代器是不会停止可以无限循环的,
        数据一遍推完后它会回到起点再接着推

        Parameters:
            batch_size (int): - 设定最大batch长度

        Yield:
            tuple[np.ndarray,np.ndarray]: - 由特征组(batch_size*n维)\
            和标签onehot(batch_size*一维)组组成的元组,特征维数要看预处理是怎么做的
        """
        batch_gen = self._batch_gen(self.TEST_DATA, batch_size)
        loopcount = next(batch_gen)
        yield loopcount
        while True:
            batch_X, batch_y = next(batch_gen)
            yield batch_X, batch_y
