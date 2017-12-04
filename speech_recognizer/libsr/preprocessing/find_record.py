from pathlib import Path
DEFAULT_DATASET_PATH = Path(__file__).absolute(
).parent.parent.parent.parent.joinpath("dataset")
TRAIN_SET = "train/audio"
TEST_SET = "test/audio"


def find_train_data(path_str=str(DEFAULT_DATASET_PATH.joinpath(TRAIN_SET))):
    p = Path(path_str)
    data = []
    for i in p.iterdir():
        clz = i.name
        for j in i.iterdir():
            item_path = str(j.absolute())
            data.append((item_path,clz))
    return data
