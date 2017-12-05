REGIST_PERPROCESS = {}


def regist(func):
    REGIST_PERPROCESS[func.__name__] = func
    return func
