"""
auxiliary functions
"""

from timeit import default_timer
from functools import wraps
from numpy import sqrt
import logging


def bisection(vector, value):
    """
    For <vector> and <value>, returns an index j such that <value> is between
    vector[j] and vector[j+1]. Values in <vector> must increase
    monotonically. j=-1 or j=len(vector) is returned to indicate that
    <value> is out of range below and above, respectively.
    ref.:
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    n = len(vector)
    if value < vector[0]:
        return -1
    elif value > vector[n - 1]:
        return n
    jl = 0  # Initialize lower
    ju = n - 1  # and upper limits.
    while ju - jl > 1:
        # If we are not yet done,
        jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
        if value >= vector[jm]:
            jl = jm  # and replace either the lower limit
        else:
            ju = jm  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if value == vector[0]:  # edge cases at bottom
        return 0
    elif value == vector[n - 1]:  # and top
        return n - 1
    else:
        return jl


def l1_minimum(x0, *args):
    """
    returns the cost function for a Lasso regression (L1 norm)
    :param x0: array - [f0, b] such that f = i * x0[0] * sqrt(1. + x0[1] * i**2)
    :param args: array - measured resonance frequencies as from FFT
    :return: float - l1 cost function
    """
    x0 = list(x0)  # x0 comes in as ndarray if invoked from the scipy minimizer
    if not args or not x0:
        return 0.
    freq = list()
    fo = [i for i in args]
    fmax = max(fo)
    for i in range(1, 640):
        f = i * x0[0] * sqrt(1. + x0[1] * i * i)
        freq.append(f)
        if f > fmax:
            break  # exit if superseded max. frequency measured to save time
    num_freq = len(freq)
    l1 = 0.  # l1 cost function
    for found in fo:
        idx = bisection(freq, found)
        # L1 norm
        if idx == -1:
            l1 += abs(found - freq[0])  # <min frequency
        elif idx == num_freq:
            l1 += abs(found - freq[num_freq - 1])  # >max frequency
        else:
            # consider the closest candidate of neighbors
            l1 += min(abs(found - freq[idx]), abs(found - freq[idx + 1]))
    return l1


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
