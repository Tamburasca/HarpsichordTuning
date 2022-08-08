"""
minimizer for L1 norm
"""

from __future__ import annotations
from numpy import sqrt, array, sign, ndarray
from numdifftools import Jacobian, Hessian
from typing import List, Tuple


def bisection(vector: List, value: float) -> int:
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


class L1(object):
    def __init__(self, ind: List):
        """
        :param ind: array - measured resonance frequencies as from peaks (FFT)
        """
        self.fo = ind
        self.fmax = max(self.fo)
        self.l1_first = None
        self.l1_last = None
        self.jacobi = array([0., 0.])

    def l1_minimum(self, x0: List, jac: bool = False) -> float:
        """
        returns the cost function for a regression on the L1 norm
        l1 = sum( abs(f_i(measured) - f_i(calculated) ) )
        :param x0: array - [f0, b] such that
            f = i * x0[0] * sqrt(1. + x0[1] * i**2)
        :param jac: bool - if jacobi is to be calculated
        :return: float - l1 cost function
        """
        freq = list()
        x0 = list(x0)  # x0 comes as ndarray if invoked from the scipy minimizer
        self.jacobi = array([0., 0.])

        for i in range(1, 640):
            f = i * x0[0] * sqrt(1. + x0[1] * i * i)
            freq.append(f)
            if f > self.fmax:
                break  # exit if superseded max. frequency measured to save time
        num_freq = len(freq)
        l1 = 0.  # l1 cost function
        for found in self.fo:
            # loop over peaks found
            idx = bisection(freq, found)
            # L1 norm
            if idx == -1:
                diff = found - freq[0]
                l1 += abs(diff)  # <min frequency
                if jac:
                    self.jacobi += \
                        self.__derivative(x0=x0, i=idx) * sign(-diff)
            elif idx == num_freq:
                diff = found - freq[num_freq - 1]
                l1 += abs(diff)  # >max frequency
                if jac:
                    self.jacobi += \
                        self.__derivative(x0=x0, i=idx) * sign(-diff)
            else:
                # consider the closest candidate of neighbors
                l1 += min(abs(found - freq[idx]), abs(found - freq[idx + 1]))
                if jac:
                    if abs(found - freq[idx]) < abs(found - freq[idx + 1]):
                        diff = found - freq[idx]
                        self.jacobi += \
                            self.__derivative(x0=x0, i=idx) * sign(-diff)
                    else:
                        diff = found - freq[idx + 1]
                        self.jacobi += \
                            self.__derivative(x0=x0, i=idx + 1) * sign(-diff)

        if self.l1_first is None:
            self.l1_first = l1
        self.l1_last = l1

        return l1

    def l1_minimum_log_b(self, x0: List, jac: bool = False) -> float:
        # b is coming in as log10(b)
        x0[1] = 10 ** x0[1]
        return self.l1_minimum(x0, jac)

    def l1_minimum_jac(self, x0: List) -> ndarray:
        return Jacobian(lambda x0: self.l1_minimum(x0))(x0).ravel()

    def l1_minimum_jac_direct(self, x0: List) -> ndarray:
        self.l1_minimum(x0, jac=True)
        return self.jacobi

    def l1_minimum_hess(self, x0: List) -> ndarray:
        return Hessian(lambda x0: self.l1_minimum(x0))(x0)

    def l1_minimum_der(self, x0: List) -> Tuple[float, ndarray]:
        return self.l1_minimum(x0, jac=True), self.jacobi

    def compare_l1(self) -> bool:
        return self.l1_last < self.l1_first

    @staticmethod
    def __derivative(x0, i):
        if x0[1] < 0.:
            x0[1] = 0.
        tmp = sqrt(1. + x0[1] * i * i)
        deriv_f0 = i * tmp
        deriv_b = 0.5 * i ** 3 * x0[0] / tmp
        return array([deriv_f0, deriv_b])
