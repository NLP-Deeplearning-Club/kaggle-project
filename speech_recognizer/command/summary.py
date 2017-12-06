from argparse import Namespace
from speech_recognizer.libsr import summary


def summary_command(args: Namespace)->None:
    print(summary(args.process_name))
