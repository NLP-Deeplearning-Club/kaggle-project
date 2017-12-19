import subprocess
from pathlib import Path
from argparse import Namespace
from speech_recognizer.conf import LOG_PATH


def run_tf_board(process_name):
    p = Path(LOG_PATH).joinpath(str(process_name))
    print(p)
    command = ["tensorboard", "--logdir=" + str(p)]
    subprocess.call(command, shell=False)


def tf_board_command(args: Namespace)->None:
    """根据命令行进行训练操作
    """
    run_tf_board(args.process_name)
