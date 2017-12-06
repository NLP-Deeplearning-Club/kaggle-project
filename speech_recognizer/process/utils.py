REGIST_PROCESS = {}
REGIST_PERPROCESS = {}


class regist:
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, func):
        REGIST_PROCESS[func.__name__] = func
        REGIST_PERPROCESS[func.__name__] = self.preprocess
        return func
