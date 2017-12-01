import numpy as np
from scipy.io import wavfile
import python_speech_features


def mfcc(audio, sample_rate, **kwargs):
    """从读出的音频数据中算出mfcc,具体可以看python_speech_features的文档

    Parameters:
        audio (np.ndarray): - 指明音频的振幅序列
        sample_rate (int): - 指明抽样率
        numcep (int): - 指明返回的倒数数量,默认为13

    Returns:
        np.ndarray: - mfcc强度(二维)组成的元组,shape为(times.shape,numcep)
    """
    return python_speech_features.mfcc(audio, sample_rate, **kwargs)


def mfcc_from_path(record_path, **kwargs):
    """从音频文件读取出mfcc

    Parameters:
        record_path (Union[pathlib.Path,str]): - 指明音频的路径

    Returns:
        np.ndarray: - mfcc强度(二维)组成的元组,shape为(times.shape,numcep)
    """
    sample_rate, samples = wavfile.read(str(record_path))
    return mfcc(samples, sample_rate,**kwargs)
