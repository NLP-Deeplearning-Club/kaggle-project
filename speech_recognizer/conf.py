from pathlib import Path

TRAIN_DATASET_PATH = str(Path(__file__).absolute(
).parent.parent.parent.parent.joinpath("dataset/train/audio"))

TEST_DATASET_PATH = str(Path(__file__).absolute(
).parent.parent.parent.parent.joinpath("dataset/test/audio"))

MODEL_PATH = str(Path(__file__).absolute(
).parent.parent.joinpath("serialized_models/"))

LOG_PATH = str(Path(__file__).parent.parent.joinpath("tmp"))

WANTED_WORDS = ['yes', 'no', 'up', 'down',
                'left', 'right', 'on', 'off', 'stop', 'go']

POSSIBLE_LABELS = ['silence', 'unknown'] + WANTED_WORDS
