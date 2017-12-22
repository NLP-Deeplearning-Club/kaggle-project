from pathlib import Path
import keras_attention_block

TRAIN_DATASET_PATH = str(Path(__file__).absolute(
).parent.parent.joinpath("dataset/train/audio"))

TEST_DATASET_PATH = str(Path(__file__).absolute(
).parent.parent.joinpath("dataset/test/audio"))

MODEL_PATH = str(Path(__file__).absolute(
).parent.joinpath("serialized_models/"))

LOG_PATH = str(Path(__file__).parent.joinpath("tmp"))

WANTED_WORDS = ['yes', 'no', 'up', 'down',
                'left', 'right', 'on', 'off', 'stop', 'go']

POSSIBLE_LABELS = ['silence', 'unknown'] + WANTED_WORDS

CUSTOM_OBJECT = {i: getattr(keras_attention_block, i)
                 for i in keras_attention_block.__dir__() if i[0].isupper()}
