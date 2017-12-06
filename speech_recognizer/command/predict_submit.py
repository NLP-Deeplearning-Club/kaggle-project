from pathlib import Path
from argparse import Namespace
from time import gmtime, strftime
from tqdm import tqdm
import numpy as np
from speech_recognizer.process.utils import (
    REGIST_PERPROCESS
)
from speech_recognizer.libsr import predict


def test_data_gen(process_name, batch_size=200):
    p = Path(__file__).absolute().parent.parent.parent.joinpath(
        "dataset/test/audio")
    preprocess = REGIST_PERPROCESS.get(process_name)
    gen = p.iterdir()
    stop = False
    while True:
        batch_X = []
        batch_y = []
        for i in range(batch_size):
            try:
                i = next(gen)
                fname = i.name
                temp = preprocess(str(i.absolute()))
            except StopIteration:
                stop = True
                break
            except Exception as e:
                print(str(e))
                break
            batch_X.append(fname)
            batch_y.append(temp)
        yield batch_X, np.array(batch_y)
        if stop:
            raise StopIteration("end")


def predict_submit_command(args: Namespace)->None:
    """根据命令行进行训练操作
    """
    p = Path(__file__).absolute().parent.parent.parent.joinpath(
        "dataset/test/audio")
    lenght = len(list(p.iterdir()))
    timenow = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    sub_name = 'submission_' + args.process_name + \
        '_' + str(timenow) + '.csv'
    sub_dir = Path("submit_files")
    if sub_dir.is_dir():
        sub_file = str(sub_dir.joinpath(sub_name))
    else:
        sub_file = sub_name
    p_save = Path(sub_file).absolute()
    with open(str(p_save), 'w') as fout:
        fout.write('fname,label\n')
        with tqdm(total=lenght) as schedule:
            batch_size = 3000
            gen = test_data_gen(args.process_name, batch_size)
            for names, X in gen:
                labels = predict(args.process_name, X,
                                 args.batch_size, args.verbose)
                for fname, label in zip(names, (lab for lab, _ in labels)):
                    fout.write('{},{}\n'.format(fname, label))
                    schedule.update(1)
