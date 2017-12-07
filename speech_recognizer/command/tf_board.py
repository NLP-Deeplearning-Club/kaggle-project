from argparse import Namespace
from speech_recognizer.process.utils import REGIST_PROCESS
from speech_recognizer.process import *
import subprocess
from pathlib import Path


def run_tf_board(process_name):
    p = Path(__file__).parent.parent.joinpath(
        "tmp/{}".format(process_name))
    print(p)
    command = ["tensorboard", "--logdir=" + str(p)]
    subprocess.call(command, shell=False)


def tf_board_command(args: Namespace)->None:
    """根据命令行进行训练操作
    """
    run_tf_board(args.process_name)
