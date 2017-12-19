from argparse import Namespace
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from speech_recognizer.process.utils import (
    REGIST_PERPROCESS,
    REGIST_FEATURE_EXTRACT
)
from .utils import predict


def predict_command(args: Namespace)->None:
    """根据命令行进行训练操作
    """
    p = Path(args.wav_dir_path)
    preprocess = REGIST_PERPROCESS.get(args.process_name)
    feature_extract = REGIST_FEATURE_EXTRACT.get(args.process_name)
    n_r_s = ((i.name, wavfile.read(str(i.absolute()))) for i in p.iterdir())
    XX = []
    for name, (rate, samples) in n_r_s:
        tr, ts = preprocess(rate, samples)
        feature = feature_extract(tr, ts)
        XX.append((name, feature))
    X = np.array([i for _, i in XX])
    pre = predict(args.process_name, X, args.batch_size, args.verbose)
    print(list(zip([i for i, _ in XX], pre)))
