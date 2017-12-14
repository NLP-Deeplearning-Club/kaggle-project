"""使wav数据整齐"""
import numpy as np


def desired_samples_wav(wav, desired_samples=16000):
    """ 如果wav数据没有采样足够,则用0值补齐否则随机的在 wav 数据里面挑选一段足够采样的连续完整wav出来

    Args:
        wav (np.ndarray) : - 一维的音频序列
        desired_samples (int) : - 要求的采样数目,默认为16000

    Returns:
        np.dnarray: -长度与desired_samples值一致的wav一维数组
    """
    wav_length = wav.shape[0]
    if wav_length < desired_samples:
        # Pad 0 at the end
        desired_wav = np.lib.pad(
            wav, (0, desired_samples - wav_length), mode='constant')
    elif wav_length > desired_samples:
        # Random choose a range from the data
        start = np.random.randint(0, wav_length - desired_samples)
        desired_wav = wav[start:start + desired_samples]
    else:
        desired_wav = wav
    return desired_wav
