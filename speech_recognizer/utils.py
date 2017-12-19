import numpy as np
from speech_recognizer.conf import (
    POSSIBLE_LABELS
)


def lab_to_vector(label):
    vector = np.array([(1 if i == label else 0) for i in POSSIBLE_LABELS])
    return vector


def vector_to_lab(vector):
    return max(zip(POSSIBLE_LABELS, vector), key=lambda x: x[1])[0]


