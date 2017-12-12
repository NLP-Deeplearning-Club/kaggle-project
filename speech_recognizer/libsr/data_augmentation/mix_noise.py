""" 随机的向 wav 数据里面加入背景噪音 """

import numpy as np
import glob
from scipy.io import wavfile
import random

def mix_background_noise(wav, background_volume_range=0.1, background_frequency=0.1):

    """ 随机的向 wav 数据里面加入噪音

    Args:
        wav (np.ndarray) : - 一维的音频序列
        background_volume_range (float) : - 背景音的缩放量，背景音一般都是最大值为1，但是我们的 audio 一般是0.1左右的最大值，所以需要缩放一下背景音
        background_frequency (float) : - 背景音出现的概率，默认为 0.1

    Returns:
        加上背景音的 wav 一维数组
    """
    
    wav_length = wav.shape[0]
    wav = wav.reshape(wav_length, 1)
    
    # Random choose a background noise
    background_samples = _pick_random_noise()
    
    if wav_length <= len(background_samples): 
        # Random shift the background noise to get the data
        background_offset = np.random.randint(0, len(background_samples) - wav_length)
        background_clipped = background_samples[background_offset:(background_offset + wav_length)]
        background_reshaped = background_clipped.reshape([wav_length, 1])
    else:
        # Pad 0 at the end if length of wav is longer than background 
        background_reshaped = np.lib.pad(background_samples, 
                                       (0, wav_length-len(background_samples)),
                                       'constant')
    
    # Random choose add background noise or not
    if np.random.uniform(0, 1) < background_frequency:
        background_volume = np.random.uniform(0, background_volume_range)
    else:
        background_volume = 0
    background_noise = np.multiply(background_reshaped, background_volume)
    wav_with_noise = background_noise + wav   
    
    # Clip by -1, 1
    background_clamp = np.clip(wav_with_noise, -1.0, 1.0)
    
    return background_clamp.reshape(wav_length)


def _pick_random_noise():
    background_wavfiles = glob.glob('../../../dataset/train/audio/_background_noise_/*.wav')
    wf = random.choice(background_wavfiles)
    return _load_wav_file(wf)


def _load_wav_file(filename):
    """Loads an audio file and returns a float PCM-encoded array of samples.
  
    Args:
        filename: Path to the .wav file to load.
    Returns:
        Numpy array holding the sample data as floats between -1.0 and 1.0.
    """
    
    _, wav = wavfile.read(str(filename))
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav

if __name__=='__main__':
    wav = _load_wav_file('../../../dataset/train/audio/right/988e2f9a_nohash_0.wav')
    print(mix_background_noise(wav, 0.1,1))
