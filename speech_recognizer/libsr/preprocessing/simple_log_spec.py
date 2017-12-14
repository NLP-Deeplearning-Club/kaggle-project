"""从路径获取数据,先进行重采样,再将结果用于计算对数频谱,最后padding后返回频谱数据
"""
from .steps.log_specgram import log_specgram
from .steps.resample import resample
from .steps.padding import padding_wave


def simple_log_spec_perprocess(sample_rate, samples, cnn=False):
    """使用对数频谱数据作为特征,预处理顺序为:
    log_specgram->添加一维用于作为图形处理

    Parameters:
        sample_rate (int): - 音频采样率
        samples (np.ndarray): - wav数据
        cnn (bool): - 是否为cnnreshape出一个维度,默认为False

    yield:
        (np.ndarray): - 若cnn参数为True.则返回的特征(3维),本处为(99, 81, 1),\
        否则返回特征(2维),本处为(99, 81)
    """
    _, _, specgram = log_specgram(
        samples, sample_rate=sample_rate)
    X_yield = specgram
    if cnn:
        X_yield = X_yield.reshape(tuple(list(X_yield.shape) + [1]))
    return X_yield
