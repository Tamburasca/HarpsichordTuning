"""
auxiliary functions
"""

from __future__ import annotations
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags, spdiags
from numpy import ones, array
from timeit import default_timer
from functools import wraps
import logging
from typing import Callable
from numpy.typing import NDArray, ArrayLike


def mytimer(supersede: Callable | str = None) -> Callable:
    """
    wrapper around function for which the consumed time is to be measured in
    DEBUG mode. Call either via
    1) mytimer,
    2) mytimer(), or
    3) mytimer("<supersede function name>")
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
    Note: currently this function is not used!
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


@mytimer("baseline calculation")
def baseline_als_optimized(y: ArrayLike,
                           lam: float,
                           p: float,
                           niter: int = 10
                           ) -> NDArray:
    """
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    Caveat: this function is disregarded for the time being since it consumes
    between 80 and 110 ms.
    """
    z, z_last = array([]), array([])
    lth = len(y)
    d = diags([1, -2, 1], [0, -1, -2], shape=(lth, lth-2))
    # Precompute this term since it does not depend on `w`
    d = lam * d.dot(d.transpose())
    w = ones(lth)
    wo = spdiags(w, 0, lth, lth)
    for i in range(niter):
        wo.setdiag(w)  # Do not create a new matrix, just update diagonal values
        zo = wo + d
        z = spsolve(zo, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
        # following early exit clause yields another 50 msec in speed
        if i > 0:
            if all(abs(z - z_last)) < 1.e-1:
                break
        z_last = z

    return z
