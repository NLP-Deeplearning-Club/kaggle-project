from argparse import Namespace
from speech_recognizer.process.utils import REGIST_PROCESS
from speech_recognizer.process import *

def train_command(args: Namespace)->None:
    """根据命令行进行训练操作
    """
    process = REGIST_PROCESS.get(args.process_name)
    process()
    
