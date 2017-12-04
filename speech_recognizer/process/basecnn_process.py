"""示范用的测试训练过程
"""
from speech_recognizer.libsr.preprocessing.find_record import find_train_data
from .utils import regist


@regist
def basecnn_process():
    print(len(find_train_data()))
    print(find_train_data()[0])