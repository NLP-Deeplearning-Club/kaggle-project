from argparse import Namespace
from speech_recognizer.libsr.preprocessing.blueprints.utils import (
    REGIST_PERPROCESS
)
from speech_recognizer.process import *
libsr.preprocessing.blueprints


def predict_submit_command(args: Namespace)->None:
    """根据命令行进行训练操作
    """
    preprocess = REGIST_PERPROCESS.get(args.preprocess_name)
    X = preprocess(args.wav_path)
