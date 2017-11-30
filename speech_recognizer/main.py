import sys
import argparse
from .command.pretreat import pretreat_command


class Command:
    """用于管理所有命令行操作指令的类
    """

    def __init__(self, argv):
        parser = argparse.ArgumentParser(
            description='语音识别器',
            usage='''speech_recognizer.py <command> [<args>]

支持的操作有:
   pretreat        预处理一段或者一个文件夹下的音频信息,并将其保存到指定文件夹下
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

    def pretreat(self):
        """预处理命令
        """
        parser = argparse.ArgumentParser(
            description='预处理一段或者一个文件夹下的音频信息,并将其保存到指定文件夹下')
        parser.add_argument("path", type=str)
        parser.add_argument("-o", "--output_dir", type=str, help="用于指定输出的文件夹")
        parser.add_argument("--toimg", action='store_true', help="用于指定输出的为频谱图")
        parser.add_argument("--tomfcc", action='store_true',
                            help="用于指定输出的为mfcc")
        parser.set_defaults(func=pretreat_command)
        args = parser.parse_args(self.argv[1:])
        args.func(args)


def main(argv=sys.argv[1:]):
    Command(argv)
