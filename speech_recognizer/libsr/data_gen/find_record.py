import math
import random
from pathlib import Path
DEFAULT_DATASET_PATH = Path(__file__).absolute(
).parent.parent.parent.parent.joinpath("dataset")
TRAIN_SET = "train/audio"
TEST_SET = "test/audio"

wanted_words = ['yes', 'no', 'up', 'down',
                'left', 'right', 'on', 'off', 'stop', 'go']
possible_labels = ['silence', 'unknown'] + wanted_words


def find_data(dataset_path=str(DEFAULT_DATASET_PATH.joinpath(TRAIN_SET)),
              validation_rate=0.1,
              test_rate=0.1,
              repeat=4,
              unknown_rate=0.1,
              silence_rate=0.1):
    """
    将训练集中的数据找出.并按比例分配为训练集和测试集.注意每一种标签都按这个比例抽样而非总体抽样

    Parameters:
        dataset_path (str): - 指明`dataset`文件夹所在路径,默认在,模块外一层
        validation_rate (float): - 指明验证集比例,这个并不是严格的,而是使用random函数随机抽取.默认为0.1
        test_rate (float): - 指明测试集比例,这个并不是严格的,而是使用random函数随机抽取.默认为0.1
        repeat (int): - 指明非unknown和非silence的数据重复多少次
        unknown_rate (float): - 指明各个数据集中未知数据的比例.默认为0.1
        silence_rate (float): - 指明各个数据集中沉默音的比例.默认为0.1

    Returns:
        tuple[list,list,list]: - 训练集(音频路径,标签),验证集(音频路径,标签)和测试集(音频路径,标签)组成的元组
    """
    p = Path(dataset_path)
    train_data = []
    test_data = []
    validation_data = []
    unknown_data = []
    silence_data = []
    for i in p.iterdir():
        clz = i.name
        if clz in wanted_words:
            for j in i.iterdir():
                item_path = str(j.absolute())
                not_train_rate = test_rate + validation_rate
                rate = random.random()
                if rate > not_train_rate:
                    train_data.append((item_path, clz))
                else:
                    if rate < validation_rate:
                        test_data.append((item_path, clz))
                    else:
                        validation_data.append((item_path, clz))
        elif clz == "_background_noise_":
            for j in i.iterdir():
                item_path = str(j.absolute())
                silence_data.append((item_path, "silence"))
        else:
            for j in i.iterdir():
                item_path = str(j.absolute())
                unknown_data.append((item_path, "unknown"))
    len_train_data = len(train_data)
    len_test_data = len(test_data)
    len_validation_data = len(validation_data)

    # 计算各个数据集中应该有多少unknown数据
    len_unknown = int(math.ceil(
        (unknown_rate / (1 - unknown_rate)) * (len_train_data + len_test_data + len_validation_data)))
    random.shuffle(unknown_data)
    samples = random.sample(unknown_data, len_unknown)
    for i in samples:
        not_train_rate = test_rate + validation_rate
        rate = random.random()
        if rate > not_train_rate:
            train_data.append(i)
        else:
            if rate < validation_rate:
                test_data.append(i)
            else:
                validation_data.append(i)

    # 计算各个数据集中应该有多少silence数据
    len_train_silence = int(math.ceil(
        (silence_rate / (1 - silence_rate)) * len_train_data))
    len_test_silence = int(math.ceil(
        (unknown_rate / (1 - silence_rate)) * len_test_data))
    len_validation_silence = int(math.ceil(
        (silence_rate / (1 - silence_rate)) * len_validation_data))
    for i in range(len_train_silence):
        train_data.append(random.choice(silence_data))
    for i in range(len_test_silence):
        test_data.append(random.choice(silence_data))
    for i in range(len_validation_silence):
        validation_data.append(random.choice(silence_data))
    random.shuffle(train_data)
    random.shuffle(validation_data)
    random.shuffle(test_data)
    return train_data, validation_data, test_data
