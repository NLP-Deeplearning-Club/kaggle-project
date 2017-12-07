from argparse import Namespace
import multiprocessing
from speech_recognizer.process.utils import REGIST_PROCESS
from speech_recognizer.process import *
from .tf_board import run_tf_board


def train_command(args: Namespace)->None:
    """根据命令行进行训练操作
    """
    p = None
    if args.tf_board:
        p = multiprocessing.Process(target=run_tf_board,args=[args.process_name])
        p.daemon = True
        p.start()
        
    process = REGIST_PROCESS.get(args.process_name)
    process()
    if p:
        p.join()
        p.terminate()
