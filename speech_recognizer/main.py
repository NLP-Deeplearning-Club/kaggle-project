import sys
import argparse
from speech_recognizer.command.train import train_command


class Command:
    """用于管理所有命令行操作指令的类
    """

    def __init__(self, argv):
        parser = argparse.ArgumentParser(
            description='语音识别器',
            usage='''speech_recognizer.py <command> [<args>]

支持的操作有:
   train        按过程名训练一个模型
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


def main(argv=sys.argv[1:]):
    Command(argv)
