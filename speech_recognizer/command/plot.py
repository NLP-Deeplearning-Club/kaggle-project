from argparse import Namespace
from speech_recognizer.libsr import plot


def plot_command(args: Namespace)->None:
    if plot:
        plot(args.process_name)
    else:
        print("需要先安装 graphviz 和 pydot-ng")
