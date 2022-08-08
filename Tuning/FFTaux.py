"""
auxiliary functions
"""

from timeit import default_timer
from functools import wraps
import logging
from typing import Callable, Union


def mytimer(supersede: Union[Callable, str] = None) -> Callable:
    """
    wrapper around function for which the consumed time is measured in
    DEBUG mode. Call either via mytimer, mytimer(), or
    mytimer("<supersede function name>")
    :param supersede: string (default=None)
    """

    def _decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = default_timer()
            result = func(*args, **kwargs)
            logging.debug("Time utilized for {}: {:.2f} ms".format(
                func.__name__ if 'supersede' not in locals()
                                 or callable(supersede)
                                 or supersede is None else supersede,
                (default_timer() - start_time) * 1_000)
            )
            return result

        return wrapper

    return _decorator(supersede) if callable(supersede) else _decorator


def log10_b(func: Callable) -> Callable:
    """
    wrapper converts the log10(b) of 1st parameter to b again to provide
    equidistant slices in log10 space
    :param func:
    :return:
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # args[0] is supposed to be x0, if provided
        if len(args) > 0:
            args[0][1] = 10. ** args[0][1]
        else:
            kwargs['x0'][1] = 10. ** kwargs['x0'][1]
        return func(self, *args, **kwargs)

    return wrapper
