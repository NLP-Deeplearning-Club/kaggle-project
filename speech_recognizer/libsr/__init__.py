from pathlib import Path
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

def predict(process_name):
    if isinstance(process_name,str):
        model = _load(process_name)
    else:
        model = process_name
    return model.predict()

def summary(process_name):
    if isinstance(process_name,str):
        model = _load(process_name)
    else:
        model = process_name
    return model.summary()

if plot_model is not None:
    def plot(process_name):
        if isinstance(process_name,str):
            model = _load(process_name)
        else:
            model = process_name
        return plot_model(model, to_file='model.png')
