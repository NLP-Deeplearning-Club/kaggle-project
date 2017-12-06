import sys
import argparse
from speech_recognizer.command.train import train_command
from speech_recognizer.command.predict import predict_command
from speech_recognizer.command.predict_submit import predict_submit_command
from speech_recognizer.command.summary import summary_command
from speech_recognizer.command.plot import plot_command
from speech_recognizer.process.utils import REGIST_PROCESS


class Command:
    """用于管理所有命令行操作指令的类
    """

    def __init__(self, argv):
        parser = argparse.ArgumentParser(
            description='语音识别器',
            usage='''speech_recognizer.py <command> [<args>]

支持的操作有:
   train             按过程名训练一个模型
   predict           按过程名,预处理过程名以及要预测的音频所在文件夹预测种类
   predict_submit    预测测试集中的数据,并生成提交用的csv文件
   summary           指定的训练过程输出模型结构

''')
        parser.add_argument('command', help='Subcommand to run')

        self.argv = argv
        args = parser.parse_args(argv[0:1])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        """训练命令
        """
        parser = argparse.ArgumentParser(
            description='训练一个模型,可选的有:{}'.format(",".join(
                list(REGIST_PROCESS.keys()))))
        parser.add_argument("process_name", type=str)
        parser.set_defaults(func=train_command)
        args = parser.parse_args(self.argv[1:])
        args.func(args)

    def predict(self):
        """预测命令"""
        parser = argparse.ArgumentParser(
            description='''使用指定的模型和预处理过程预测一个文件夹下的音频,
可选的模型有:{}'''.format(",".join(
                list(REGIST_PROCESS.keys()))))
        parser.add_argument("process_name", type=str)
        parser.add_argument("wav_dir_path", type=str)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--verbose", type=int, default=0)
        parser.set_defaults(func=predict_command)
        args = parser.parse_args(self.argv[1:])
        args.func(args)

    def predict_submit(self):
        """预测命令"""
        parser = argparse.ArgumentParser(
            description='''使用指定的模型和预处理过程预测测试文件夹下的音频,并输出为标准格式的文件,
可选的模型有:{}'''.format(",".join(
                list(REGIST_PROCESS.keys()))))
        parser.add_argument("process_name", type=str)
        parser.add_argument('-s', "--size", type=int, default=30000)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--verbose", type=int, default=0)
        parser.set_defaults(func=predict_submit_command)
        args = parser.parse_args(self.argv[1:])
        args.func(args)

    def summary(self):
        parser = argparse.ArgumentParser(
            description='''指定的训练过程输出模型结构
可选的模型有:{}'''.format(",".join(
                list(REGIST_PROCESS.keys()))))
        parser.add_argument("process_name", type=str)
        parser.set_defaults(func=summary_command)
        args = parser.parse_args(self.argv[1:])
        args.func(args)

    def plot(self):
        parser = argparse.ArgumentParser(
            description='''指定的训练过程输出模型结构图像
可选的模型有:{}'''.format(",".join(
                list(REGIST_PROCESS.keys()))))
        parser.add_argument("process_name", type=str)
        parser.set_defaults(func=plot_command)
        args = parser.parse_args(self.argv[1:])
        args.func(args)


def main(argv=sys.argv[1:]):
    Command(argv)
