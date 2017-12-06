import sys
import argparse
from speech_recognizer.command.train import train_command
from speech_recognizer.command.predict import predict_command
from speech_recognizer.command.predict_submit import predict_submit_command

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
            description='训练一个模型')
        parser.add_argument("process_name", type=str)
        parser.set_defaults(func=train_command)
        args = parser.parse_args(self.argv[1:])
        args.func(args)

    def predict(self):
        """预测命令"""
        parser = argparse.ArgumentParser(
            description='训练一个模型')
        parser.add_argument("process_name", type=str)
        parser.add_argument("preprocess_name", type=str)
        parser.add_argument("wav_dir_path", type=str)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--verbose", type=int, default=0)
        parser.set_defaults(func=predict_command)
        args = parser.parse_args(self.argv[1:])
        args.func(args)

    def predict_submit(self):
        """预测命令"""
        parser = argparse.ArgumentParser(
            description='训练一个模型')
        parser.add_argument("process_name", type=str)
        parser.add_argument("preprocess_name", type=str)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--verbose", type=int, default=0)
        parser.set_defaults(func=predict_submit_command)
        args = parser.parse_args(self.argv[1:])
        args.func(args)


def main(argv=sys.argv[1:]):
    Command(argv)
