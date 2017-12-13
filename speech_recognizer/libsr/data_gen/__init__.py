from functools import partial
from speech_recognizer.libsr.preprocessing import (
    log_spec_perprocess
)
from speech_recognizer.libsr.preprocessing.mfcc_perprocess import (
    mfcc_perprocess
)
from .utils import (
    get_train_data,
    train_gen,
    get_validation_data,
    validation_gen,
    get_test_data,
    test_gen
)

log_spec_train_gen = partial(train_gen, log_spec_perprocess)
log_spec_train_data = partial(get_train_data, log_spec_perprocess)
log_spec_validation_gen = partial(validation_gen, log_spec_perprocess)
log_spec_validation_data = partial(get_validation_data, log_spec_perprocess)
log_spec_test_gen = partial(test_gen, log_spec_perprocess)
log_spec_test_data = partial(get_test_data, log_spec_perprocess)

mfcc_train_gen = partial(train_gen, mfcc_perprocess)
mfcc_train_data = partial(get_train_data, mfcc_perprocess)
mfcc_validation_gen = partial(validation_gen, mfcc_perprocess)
mfcc_validation_data = partial(get_validation_data, mfcc_perprocess)
mfcc_test_gen = partial(test_gen, mfcc_perprocess)
mfcc_test_data = partial(get_test_data, mfcc_perprocess)
