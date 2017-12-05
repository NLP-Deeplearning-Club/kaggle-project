"""去噪音模块
"""

from scipy.io import wavfile
from scipy import signal
import re
from glob import glob
import os
import numpy as np

def get_main_voice(wav, filter_threshold=0.1, seg_length=800):
    """ Keep only the main voice in wav
    
    Args:
        wav: The wav data to be processed, in a numeric series list
        filter_threshold: Threshold to be considered as no signal
        seg_length: How long is the segment that we will cut the wav for filtering
        
    Returns:
        wav with only the main voice kept
    """
    
    # Split wav into segements
    splitted_wavs = _split_by_length(wav, seg_length)
    
    # Compute the avg of all segments
    wavs_mean = []
    for sw in splitted_wavs:
        wavs_mean.append(np.absolute(sw).mean())
    wavs_mean = np.array(wavs_mean)

    # Check if each segments will be kept or not
    # seg_keep_array : [1 0 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 1 0]
    seg_keep_array = (wavs_mean > filter_threshold * wavs_mean.max()).astype(int)

    # Find the longgest keep period
    # [1 0 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 1 0] -> 5, 8 (longest continunous 1)
    start, end = _get_longest_period(seg_keep_array)
    if start > 1:
        start = (start-1)
    if end < num_seg -1:
        end = (end+1)

    kept_wav = wav[int(seg_length*start):int(seg_length*end)]
    return kept_wav

##############################################################################

def _split_by_length(wav, seg_length):
    """Split the wav by segment length"""
    
    wav_length = wav.shape[0]
    num_seg = int( full_wav_length / seg_length ) + 1
    
    indice = 0
    splitted_wavs = []
    
    while indice + seg_length <= wav_length:
        w = wav[indice: indice+seg_length]
        splitted_wavs.append(w)
        indice += seg_length

    return np.array(splitted_wavs)

##############################################################################

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

##############################################################################

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