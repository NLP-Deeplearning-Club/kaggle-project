"""先进行重采样,再将结果用于计算对数频谱,最后padding后返回频谱数据
"""
import numpy as np
from scipy.io import wavfile
from ..find_record import find_train_data
from ..resample import resample
from ..padding import padding_wave


ALL_TRAIN_DATA = find_train_data()

np.random.shuffle(ALL_TRAIN_DATA)


def log_spec_gen(batch_size=50):
    ylen = len(ALL_TRAIN_DATA)
    loopcount = ylen // batch_size
    X = []
    y = []
    for f, t in ALL_TRAIN_DATA:
        X.append(f)
        y.append(t)

    while True:
        i = np.random.randint(0, loopcount)
        _X_yield = []
        _y_yield = []
        for i, label in zip(X[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]):
            sample_rate, resampled = wavfile.read(i)
            samples = padding_wave(samples)
            if len(samples) > 16000:
                n_samples = chop_audio(samples)
            else:
                n_samples = [samples]
            for samples in n_samples:
                new_sample_rate, resampled = resample(samples))
                _, _, specgram=log_specgram(
                    resampled, sample_rate = new_sample_rate)
                _X_yield.append(specgram)
                _y_yield.append(label)
        X_yield=np.array(_X_yield)
        y_yield=label_transform(_y_yield)
        label_index=y_yield.columns.values
        y_yield=y_yield.values
        y_yield=np.array(y_yield)
        yield X_yield, y_yield


def test_data_generator(batch=16):
    fpaths = glob(os.path.join(test_data_path, '*wav'))
    i = 0
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        i += 1
        rate, samples = wavfile.read(path)
        samples = pad_audio(samples)
        resampled = signal.resample(samples, int(new_sample_rate / rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        imgs.append(specgram)
        fnames.append(path.split('\\')[-1])
        if i == batch:
            i = 0
            imgs = np.array(imgs)
            imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
            yield fnames, imgs
    if i < batch:
        imgs = np.array(imgs)
        imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
        yield fnames, imgs
    raise StopIteration()