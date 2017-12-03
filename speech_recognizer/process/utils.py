REGIST_PROCESS = []


def regist(func):
    REGIST_PROCESS.append(func.__name__)
    return func
