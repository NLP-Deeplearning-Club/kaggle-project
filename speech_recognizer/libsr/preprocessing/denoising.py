"""去噪音模块
"""

from scipy.io import wavfile
from scipy import signal
import re
from glob import glob
import os
import numpy as np


def _group_consecutives(vals, step=0):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def _get_longest_period(seg_keep_array):
    """获取最长的主干周期"""
    cons_wav = _group_consecutives(seg_keep_array)
    cons_wav_length = [sum(wav) for wav in cons_wav]
    start = 0
    end = 0
    for i in range(cons_wav_length.index(max(cons_wav_length)) + 1):
        if i < cons_wav_length.index(max(cons_wav_length)):
            start += len(cons_wav[i])
        end += len(cons_wav[i])
    return start, end


# def get_main_voice(wav, *,filter_threshold=0.1, window=num_seg):
#     """用于获取主干声音部分

#     Parameters:
#         wav (np.ndarray): - 指明音频的振幅序列
#         filter_threshold (float): - 过滤器的阈值
#         window (int): - 窗口大小,单位为采样次数(以8000hz为例,10ms窗口就是80)
#         num_seg (int): 

#     Returns:
#         np.ndarray: - 主干波形
#     """
#     """ Keep only the main voice in wav

#     Args:
#         wav: The wav data to be processed, in a numeric series list
#         filter_threshold: Threshold to be considered as no signal
#         num_seg: How many segments we will cut the wav for filtering

#     Returns:
#         wav with only the main voice kept
#     """

#     wav = signal.resample(wav, 16000)
#     seg_length = 16000 / num_seg
#     splitted_wavs = np.split(wav, num_seg)
#     wavs_mean = []
#     for sw in splitted_wavs:
#         wavs_mean.append(np.absolute(sw).mean())
#     wavs_mean = np.array(wavs_mean)
#     seg_keep_array = (wavs_mean > 0.1 * wavs_mean.max()).astype(int)

#     # Find the longgest keep period
#     # [1 0 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 1 0] -> 5, 8 (longest continunous 1)
#     start, end = _get_longest_period(seg_keep_array)
#     if start > 1:
#         start = (start - 1)
#     if end < num_seg - 1:
#         end = (end + 1)

#     keeped_wav = wav[int(seg_length * start):int(seg_length * end)]
#     return keeped_wav
