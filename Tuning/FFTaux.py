"""
auxiliary functions
"""

from timeit import default_timer
from Tuning import parameters
from functools import wraps
import logging

logging.basicConfig(format=parameters.myformat,
                    level=logging.INFO,
                    datefmt="%H:%M:%S")
if parameters.DEBUG:
    logging.getLogger().setLevel(logging.DEBUG)


def mytimer(supersede=None):
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
