from pathlib import Path
from argparse import Namespace
from collections import Counter
from time import gmtime, strftime
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.io import wavfile
from speech_recognizer.process.utils import (
    REGIST_PERPROCESS,
    REGIST_FEATURE_EXTRACT
)
from speech_recognizer.conf import TEST_DATASET_PATH
from .utils import predict


def test_data_gen(process_name, batch_size=200):
    """根据不同的过程名将数据在迭代器中进行对应的预处理从而减小内存消耗,之后再整合后按batch长度输出.从而减小内存压力.
    第一个next会返回batch_size下需要多少个迭代才能将测试集的数据遍历一遍.

    Parameters:
        process_name (str): - 寻找过程对应预处理过程的字段
        batch_size (int): - 测试数据每个batch长度

    yield:
        tuple[list,np.ndarray]: - 由文件名组(batch_size*1维)和特征组(batch_size*n维)组成的元组,\
        特征维数要看预处理是怎么做的
    """
    p = Path(__file__).absolute().parent.parent.parent.joinpath(
        "dataset/test/audio")
    preprocesses = {i: REGIST_PERPROCESS.get(i) for i in process_name}
    feature_extracts = {i: REGIST_FEATURE_EXTRACT.get(i) for i in process_name}
    gen = p.iterdir()
    stop = False
    while True:
        batch_X = []
        batch_y = []
        for i in range(batch_size):
            try:
                i = next(gen)
                fname = i.name
                rate, samples = wavfile.read(str(i.absolute()))
                temp = {}
                for p, f in preprocesses.items():
                    r, t = f(rate, samples)
                    temp[p] = feature_extracts.get(p)(r, t)
            except StopIteration:
                stop = True
                break
            except Exception as e:
                print(str(e))
                break
            batch_X.append(fname)
            batch_y.append(temp)
        table = pd.DataFrame(batch_y)
        keys = list(table.columns)
        yield batch_X, {i: np.array([j for j in table[i]]) for i in keys}
        if stop:
            raise StopIteration("end")


def predict_submit_command(args: Namespace)->None:
    """根据命令行进行训练操作
    """
    p = Path(TEST_DATASET_PATH).absolute()
    lenght = len(list(p.iterdir()))
    timenow = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    sub_name = 'submission_' + "_".join(args.process_name) + \
        '_' + str(timenow) + '.csv'

    sub_file = sub_name
    p_save = Path(sub_file).absolute()
    print("preparing data...")
    batch_size = args.size
    gen = test_data_gen(args.process_name, batch_size)
    print("preparing data done")
    with open(str(p_save), 'w') as fout:
        print("predict...")
        fout.write('fname,label\n')
        with tqdm(total=lenght) as schedule:
            for names, X in gen:
                lab_zip_list = []
                for i in args.process_name:
                    labels = predict(
                        i, X[i],
                        args.batch_size, 
                        args.verbose)
                    lab_zip_list.append(labels)
                if len(lab_zip_list) == 1:
                    labels = lab_zip_list[0]
                else:
                    labels = [max(
                        dict(Counter(sorted(i))).items(),
                        key=lambda x:x[1])[0] for i in zip(*lab_zip_list)]
                for fname, label in zip(names, labels):
                    fout.write('{},{}\n'.format(fname, label))
                    schedule.update(1)
