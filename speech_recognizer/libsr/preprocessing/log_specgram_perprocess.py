"""从路径获取数据,先进行重采样,再将结果用于计算对数频谱,最后padding后返回频谱数据
"""
from scipy.io import wavfile
from .steps.log_specgram import log_specgram
from .steps.resample import resample
from .steps.padding import padding_wave


def log_spec_perprocess(path):
    """使用对数频谱数据作为特征,预处理顺序为:
    padding->截断->resample->log_specgram->添加一维用于作为图形处理

    Parameters:
        path (file path): - 进行预处理的对象路径

    yield:
        (np.ndarray): - 返回的特征(3维),本处为(99, 81, 1)
    """
    sample_rate, samples = wavfile.read(path)
    samples = padding_wave(samples)
    if len(samples) > 16000:
        samples = samples[:16000]
    else:
        samples = samples
    new_sample_rate, resampled = resample(samples, sample_rate)
    _, _, specgram = log_specgram(
        resampled, sample_rate=new_sample_rate)
    X_yield = specgram
    X_yield = X_yield.reshape(tuple(list(X_yield.shape) + [1]))
    return X_yield
