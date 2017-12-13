from pathlib import Path
import random
DEFAULT_DATASET_PATH = Path(__file__).absolute(
).parent.parent.parent.parent.joinpath("dataset")
TRAIN_SET = "train/audio"
TEST_SET = "test/audio"


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
    for i in p.iterdir():
        clz = i.name
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

    return train_data, validation_data, test_data
