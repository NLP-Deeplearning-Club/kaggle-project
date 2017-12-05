from pathlib import Path
import json
from keras.models import load_model
from speech_recognizer.process.utils import REGIST_PROCESS
try:
    from keras.utils import plot_model
except:
    plot_model = None


def _load(process_name):
    if process_name not in REGIST_PROCESS:
        raise AttributeError("unknown process name!")
    p = Path(__file__).absolute()
    _dir = p.parent.parent
    path = _dir.joinpath("serialized_models/" + process_name + "_model.h5")
    if not path.exists():
        raise AttributeError(
            "there is no model for {process_name}!".format(process_name))
    else:
        return load_model(str(path))


def _load_index(process_name):
    if process_name not in REGIST_PROCESS:
        raise AttributeError("unknown process name!")
    p = Path(__file__).absolute()
    _dir = p.parent.parent
    path = _dir.joinpath("serialized_models/" + process_name + "_index.json")
    if not path.exists():
        raise AttributeError(
            "there is no model for {process_name}!".format(process_name))
    else:
        with open(str(path)) as f:
            result = json.load(f)
        return result


def predict(process_name, featureset, batch_size=32, verbose=0):
    if isinstance(process_name, str):
        model = _load(process_name)
    else:
        model = process_name
    pre = model.predict(featureset, batch_size=batch_size, verbose=verbose)
    labels = _load_index(process_name)

    result = [max(zip(labels, i), key=lambda x:x[1]) for i in pre]
    return result


def summary(process_name):
    if isinstance(process_name, str):
        model = _load(process_name)
    else:
        model = process_name
    return model.summary()


if plot_model is not None:
    def plot(process_name):
        if isinstance(process_name, str):
            model = _load(process_name)
        else:
            model = process_name
        return plot_model(model, to_file='model.png')
