"""从路径获取数据,先进行重采样,再将结果用于计算对数频谱,最后padding后返回mfcc数据
"""
from .steps.mfcc import mfcc
from .steps.resample import resample
from .steps.padding import padding_wave


def mfcc_perprocess(sample_rate, samples):
    """使用mfcc数据作为特征,预处理顺序为:
    padding->截断->resample->mfcc->添加一维用于作为图形处理

    Parameters:
        sample_rate (int): - 音频采样率
        samples (np.ndarray): - wav数据

    yield:
        (np.ndarray): - 返回的特征(3维),本处为(99, 13, 1)
    """
    samples = padding_wave(samples)
    if len(samples) > 16000:
        samples = samples[:16000]
    else:
        samples = samples
    new_sample_rate, resampled = resample(samples, sample_rate)
    specgram = mfcc(
        resampled, sample_rate=new_sample_rate)
    X_yield = specgram
    X_yield = X_yield.reshape(tuple(list(X_yield.shape) + [1]))
    return X_yield
