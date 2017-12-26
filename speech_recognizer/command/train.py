from argparse import Namespace
import json
import multiprocessing
from speech_recognizer.process.utils import REGIST_PROCESS
from .tf_board import run_tf_board


def _train_single(process, number, kwargs=None):
    result = []
    print()
    for i in range(number):
        print("model {}".format(i))
        if kwargs:
            res = process(**kwargs)
        else:
            res = process()
        result.append(res)
    return result


def train_command(args: Namespace)->None:
    """根据命令行进行训练操作
    """
    p = None
    if args.tf_board:
        p = multiprocessing.Process(
            target=run_tf_board,
            args=[args.process_name])
        p.daemon = True
        p.start()

    process = REGIST_PROCESS.get(args.process_name)
    kwargs = None
    if args.use_config:
        kwargs = json.load(args.use_config)
        args.use_config.close()
    number = args.number
    result = _train_single(process, number, kwargs)
    if p:
        p.terminate()
    return result
