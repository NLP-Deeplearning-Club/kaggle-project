from functools import partial
from .log_specgram_perprocess import log_spec_perprocess
from .utils import (
    get_train_data,
    get_test_data,
    train_gen,
    test_gen,
)


log_spec_train_gen = partial(train_gen, log_spec_perprocess)
log_spec_test_gen = partial(test_gen, log_spec_perprocess)
log_spec_train_data = partial(get_train_data, log_spec_perprocess)
log_spec_test_data = partial(get_test_data, log_spec_perprocess)
