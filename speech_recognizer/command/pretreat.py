from argparse import Namespace


def pretreat_command(args: Namespace)->None:
    """根据命令行进行预处理的指定操作
    """
    print(args)
