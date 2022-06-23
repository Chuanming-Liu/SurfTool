"""
Show time elapsed.

https://stackoverflow.com/a/12344609/8877268
"""
import atexit
from time import time, strftime, localtime
from datetime import timedelta


def _secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))


def _log(s, elapsed=None):
    line = "="*40
    print(line)
    print(_secondsToStr(), '-', s)
    if elapsed:
        print("Elapsed time:", elapsed)
    print(line)
    print("")

    return


def _endlog(start):
    end = time()
    elapsed = end - start
    _log("End Program", _secondsToStr(elapsed))

    return


def timing():
    """
    Show time elapsed.
    """
    start = time()
    atexit.register(_endlog, start)
    _log("Start Program")

    return
