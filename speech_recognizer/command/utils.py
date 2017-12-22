from pathlib import Path
from keras.models import load_model
from speech_recognizer.conf import MODEL_PATH, CUSTOM_OBJECT
from speech_recognizer.utils import vector_to_lab
from speech_recognizer.process.utils import REGIST_PROCESS


def _load(process_name):
    """使用process_name找到对应的模型对象
    Parameters:
        process_name (str): - 过程函数名

    Raise:
        AttributeError : - 如果过程名不在册,或者默认路径不存在,则抛出异常

    Return:
        (model): process_name对应的模型
    """
    if process_name not in REGIST_PROCESS:
        raise AttributeError("unknown process name!")
    _dir = Path(MODEL_PATH)
    path = _dir.joinpath(process_name + "_model.h5")
    if not path.exists():
        raise AttributeError(
            "there is no model for {process_name}!".format(process_name=process_name))
    else:
        return load_model(str(path), custom_objects=CUSTOM_OBJECT)


def predict(process_name, featureset, batch_size=32, verbose=0):
    """指定过程名和特征集预测标签

    Parameters:
        process_name (str): - 过程函数名,确定用什么模型进行预测
        featureset (np.ndarray)：- 由经过预处理后的特征组成的集合
        batch_size (int): - 每次训练一个batch,这个batch的大小
        verbose (int): - 预测执行时的log可见度

    Return:
        (List[Tuple[label,probability]]): 由featureset长度个(最大可能性标签,最大可能性)对组成的list
    """
    if isinstance(process_name, str):
        model = _load(process_name)
    else:
        model = process_name
    pre = model.predict(featureset, batch_size=batch_size, verbose=verbose)
    result = [vector_to_lab(i) for i in pre]
    return result


def summary(process_name):
    """指定过程名获取其summary信息

    Parameters:
        process_name (str): - 过程函数名

    Return:
        model.summary: - 模型的summary信息
    """
    if isinstance(process_name, str):
        model = _load(process_name)
    else:
        model = process_name
    return model.summary()
