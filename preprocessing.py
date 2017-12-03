from scipy.io import wavfile
from scipy import signal
import re
from glob import glob
import os
import numpy as np

##########################################################

def get_main_voice(wav, filter_threshold=0.1, num_seg=20):
    """ Keep only the main voice in wav
    
    Args:
        wav: The wav data to be processed, in a numeric series list
        filter_threshold: Threshold to be considered as no signal
        num_seg: How many segments we will cut the wav for filtering
        
    Returns:
        wav with only the main voice kept
    """
    
    wav = signal.resample(wav, 16000)
    
    seg_length = 16000 / num_seg
    # Split wav into segements
    splitted_wavs = np.split(wav, num_seg)
    # Compute the avg of all segments
    wavs_mean = []
    for sw in splitted_wavs:
        wavs_mean.append(np.absolute(sw).mean())
    wavs_mean = np.array(wavs_mean)

    # Check if each segments will be kept or not
    # seg_keep_array : [1 0 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 1 0]
    seg_keep_array = (wavs_mean > 0.1 * wavs_mean.max()).astype(int)

    def group_consecutives(vals, step=0):
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

    def get_longest_period(seg_keep_array):
        cons_wav = group_consecutives(seg_keep_array)
        cons_wav_length = [sum(wav) for wav in cons_wav]
        start = 0
        end = 0
        for i in range(cons_wav_length.index(max(cons_wav_length))+1):
            if i < cons_wav_length.index(max(cons_wav_length)):
                start += len(cons_wav[i]) 
            end += len(cons_wav[i]) 
        return start, end

    # Find the longgest keep period
    # [1 0 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 1 0] -> 5, 8 (longest continunous 1)
    start, end = get_longest_period(seg_keep_array)
    if start > 1:
        start = (start-1)
    if end < num_seg -1:
        end = (end+1)

    keeped_wav = wav[int(seg_length*start):int(seg_length*end)]
    return keeped_wav

##########################################################

def remove_muted(wav, muted_rate=0.05):
    """ Remove muted periods at the begining or end
    
    Args:
        wav: data to be processed, a numeric series
        muted_rate: under which the voice will be considered as muted,
            threshold = max_voice * muted_rate
            
    Returns:
        wav without muted part at the begining or end
    """
    
    max_voice = wav.max()
    threshold = max_voice * muted_rate
    
    keep_wav = np.array(np.absolute(wav)>threshold)
    reverse_keep_wav = np.flip(keep_wav, 0)
    start = list(keep_wav).index(True)
    end = wav.size - list(reverse_keep_wav).index(True)
    
    return wav[start:end]

##########################################################

def re_sample(wav, sample_size=4000):
    """ Re sample wav data
    
    Args:
        wav: wav data to be processed, a numeric series
        sample_size: number of samples of final returned wav
        
    Returns:
        sampled wav data
    """
    return signal.resample(wav, sample_size)