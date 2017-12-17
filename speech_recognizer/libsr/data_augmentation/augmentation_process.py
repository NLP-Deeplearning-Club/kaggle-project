from .steps.mix_noise import mix_background_noise
from .steps.time_shift import shift_and_pad_zeros
from .steps.desired_samples import desired_samples_wav
from .steps.silence_as_zero import silence_as_zero
from .steps.normalize_wav import normalize_wav

def aug_process(wav, label, mode,
                desired_samples=16000,
                time_shift=2000,
                background_volume_range=0.1,
                background_frequency=0.1):

    wav = normalize_wav(wav)
    wav = desired_samples_wav(wav, desired_samples=desired_samples)
    if mode == 'train':
        wav = silence_as_zero(wav, label)
        wav = shift_and_pad_zeros(wav, time_shift=time_shift)
    wav = mix_background_noise(wav,
                               background_volume_range=background_volume_range,
                               background_frequency=background_frequency)
    return wav
