import inspect
import json
import time
from functools import wraps
from pathlib import Path
import keras
from sklearn.metrics import (
    classification_report,
    precision_score,
    accuracy_score,
    confusion_matrix
)
from speech_recognizer.conf import MODEL_PATH, LOG_PATH, POSSIBLE_LABELS
from speech_recognizer.utils import vector_to_lab

REGIST_PROCESS = {}
REGIST_PERPROCESS = {}
REGIST_FEATURE_EXTRACT = {}


def tb_callback(process_name, histogram_freq=0,
                write_graph=True, write_images=True,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None):
    """tensorboad的回调函数,封装一层主要是为了统一log数据的保存路径

    Parameters:
        process_name (str): - 过程函数名,用于判断log数据放在什么位置
        histogram_freq (int)：- 计算各个层激活值直方图的频率（每多少个epoch计算一次），如果设置为0则不计算。
        write_graph (bool): - 是否在Tensorboard上可视化图，当设为True时，log文件可能会很大
        write_images (bool): - 是否将模型权重以图片的形式可视化
        embeddings_freq (int): - 依据该频率(以epoch为单位)筛选保存的embedding层
        embeddings_layer_names (List): - 要观察的层名称的列表，若设置为None或空列表，\
        则所有embedding层都将被观察.
        embeddings_metadata (Dict): - 将层名称映射为包含该embedding层元数据的文件名，\
        参考这里获得元数据文件格式的细节.如果所有的embedding层都使用相同的元数据文件，则可传递字符串。

    Return:
        (function): 回调函数
    """
    return keras.callbacks.TensorBoard(
        log_dir=str(Path(LOG_PATH).joinpath(process_name)),
        histogram_freq=histogram_freq,
        write_graph=write_graph, write_images=write_images,
        embeddings_freq=embeddings_freq,
        embeddings_layer_names=embeddings_layer_names,
        embeddings_metadata=embeddings_metadata
    )


def get_current_function_name():
    """获取调用的该函数的函数名


    Return:
        (str): 调用的该函数的函数名
    """
    return inspect.stack()[1][3]


class regist:
    """注册过程名到过程函数和预处理过程函数的装饰器,除了装饰后将预处理和特征提取注册起来外,\
    如果注册的函数会返回模型,那么会将这个模型保存到`<过程名>_model.h5_<时间戳整数>`的位置.\
    如果返回的除了模型还有一个准确度,那么将会把模型的信息包括训练时间,测试集上的准确度,训练时传入的参数等\
    保存到一个json文件.注意使用默认值训练的话不会保存对应参数

    Attributes:
        preprocess (callable): - 注册过程名字对应的预处理函数
    """

    def __init__(self, preprocess, feature_extract):
        self.preprocess = preprocess
        self.feature_extract = feature_extract

    def __call__(self, func):
        REGIST_PERPROCESS[func.__name__] = self.preprocess
        REGIST_FEATURE_EXTRACT[func.__name__] = self.feature_extract
        _dir = Path(MODEL_PATH)
        train_time = int(time.time())
        path = _dir.joinpath(func.__name__ + "_model.h5" + "_" +
                             str(train_time))

        @wraps(func)
        def wrapper(*args, **kwargs):
            _dir = Path(MODEL_PATH)
            train_time = int(time.time())
            path = _dir.joinpath(
                func.__name__ + "_model.h5" + "_" + str(
                    train_time))
            trained_model = func(*args, **kwargs)
            if len(trained_model) == 1:
                trained_model.save(str(path))
                print("model save done!")
                return True
            elif len(trained_model) == 2:
                model, acc = trained_model
                model.save(str(path))
                print("model save done!")
                time_str = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(train_time))
                train_info = []
                if _dir.joinpath("train_info.json").exists():
                    with open(str(_dir.joinpath("train_info.json")), "r") as f:
                        train_info = json.load(f)
                with open(str(_dir.joinpath("train_info.json")), "w") as f:
                    temp = {
                        "time": time_str,
                        "process": func.__name__,
                        "model_path": str(path),
                        "acc": acc
                    }
                    temp.update(**kwargs)
                    train_info.append(temp)
                    json.dump(train_info, f)
                return acc
            else:
                print("no model returned")
                return False
        REGIST_PROCESS[func.__name__] = wrapper
        return wrapper


def model_report(trained_model, test_data, average='macro'):
    test_datas, test_label_vectors = test_data
    pre_lab = [vector_to_lab(i) for i in trained_model.predict(test_datas)]
    lab = [vector_to_lab(i) for i in test_label_vectors]
    acc = accuracy_score(lab, pre_lab)
    print("acc: {}".format(acc))
    print("预测得分: {}".format(precision_score(lab, pre_lab, average=average)))
    print("效果报告:")
    print(classification_report(lab, pre_lab))
    print("混合矩阵:从上到下为真实标签,从左到右为预测标签,其顺序为:\n{}".format(POSSIBLE_LABELS))
    print(confusion_matrix(lab, pre_lab, labels=POSSIBLE_LABELS))
    return acc
