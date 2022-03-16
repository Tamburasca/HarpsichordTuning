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


def mytimer(superseed=None):
    """
    wrapper around function for which the consumed time is measured in
    DEBUG mode. Call either via mytimer, mytimer(), or
    mytimer("<superseed function name>")
    :param superseed: string (default=None)
    """

    def _decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = default_timer()
            result = func(*args, **kwargs)
            logging.debug("Time utilized for {}: {:.2f} ms".format(
                func.__name__ if 'superseed' not in locals()
                                 or callable(superseed)
                                 or superseed is None else superseed,
                (default_timer() - start_time) * 1_000)
            )
            return result

        return wrapper

    return _decorator(superseed) if callable(superseed) else _decorator
