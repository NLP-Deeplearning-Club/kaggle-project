import augmentation_process
from pathlib import Path
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':
    # Raed wav
    DEFAULT_DATASET_PATH = Path(__file__).absolute(
         ).parent.parent.parent.parent.joinpath('dataset')
    wav_filepath = DEFAULT_DATASET_PATH.joinpath('train', 
         'audio', 'bed', '0a7c2a8d_nohash_0.wav')
    _, wav = wavfile.read(str(wav_filepath))

    # Plot original wav
    plt.figure()
    plt.plot(wav.astype(np.float32) / np.iinfo(np.int16).max)
    plt.title('Original Wav')

    # Process with data augmen
    wav = augmentation_process.aug_process(wav, 
                                           label='bed', mode='train')

    plt.figure()
    plt.plot(wav)
    plt.title('Wav with data augmentation')

    plt.show()