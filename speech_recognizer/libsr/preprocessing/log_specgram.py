"""计算对数频谱的模块
"""
import numpy as np
from scipy import signal
from scipy.io import wavfile


def log_specgram(audio, sample_rate, *, window='hann', window_size=20,
                 step_size=10, eps=1e-10):
    """从读出的音频数据中算出对数频谱数据

    Parameters:
        audio (np.ndarray): - 指明音频的振幅序列
        sample_rate (int): - 指明抽样率
        window (str): - 指明分窗的算法,可选的详情可以看scipy.signal.get_window的文档
        window_size (Union[pathlib.Path,str]): - 指明音频的分窗大小
        step_size (Union[pathlib.Path,str]): - 指明步进长度
        eps (float): - 指明频谱强度取对数时的最小值,防止输入为0后得到负无穷

    Returns:
        tuple[np.ndarray,np.ndarray,np.ndarray]: - 由频率(一维),分段时间(一维)和频谱强度(二维)组成的元组,shape(times.shape,freqs.shape)
    """
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                            fs=sample_rate,
                                            window=window,
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


def log_specgram_from_path(record_path, *, window='hann', window_size=20,
                           step_size=10, eps=1e-10):
    """从音频文件读取出对数频谱数据

    Parameters:
        record_path (Union[pathlib.Path,str]): - 指明音频的路径
        window (str): - 指明分窗的算法,可选的详情可以看scipy.signal.get_window的文档
        window_size (Union[pathlib.Path,str]): - 指明音频的分窗大小
        step_size (Union[pathlib.Path,str]): - 指明步进长度
        eps (float): - 指明频谱强度取对数时的最小值,防止输入为0后得到负无穷

    Returns:
        tuple[np.ndarray,np.ndarray,np.ndarray]: - 由频率(一维),分段时间(一维)和频谱强度(二维)组成的元组,shape(times.shape,freqs.shape)
    """
    sample_rate, samples = wavfile.read(str(record_path))
    freqs, times, log_spec = log_specgram(samples, sample_rate,
                                          window=window,
                                          window_size=window_size,
                                          step_size=step_size, eps=eps)
    return freqs, times, log_spec
