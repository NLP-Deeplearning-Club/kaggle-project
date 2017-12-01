import numpy as np


def padding(spec, max_lenght):
    return np.pad(spec, ((0, 0), (0, 80 - len(spec[0]))), mode='constant', constant_values=0)
