REGIST_PROCESS = {}


def regist(func):
    REGIST_PROCESS[func.__name__] = func
    return func
