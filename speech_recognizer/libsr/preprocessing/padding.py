"""
"""
import numpy as np


def padding(spec, max_lenght):
    result = np.pad(spec, ((0, max_lenght - len(spec)), (0, 0)),
                    mode='constant',
                    constant_values=0)
    return result
