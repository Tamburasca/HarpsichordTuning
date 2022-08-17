"""
minimizer for L2 norm

Note: for simplicity we kept the variable names from the L1 minimizer, such that
only the class needs to be modified from L1 to L2
"""

from __future__ import annotations
from numpy import sqrt, array, ndarray
from numdifftools import Jacobian, Hessian
from typing import List, Tuple


class L2(object):
    def __init__(self, ind: List[Tuple]):
        """
        :param ind: array - measured resonance frequencies as from peaks (FFT)
        """
        self.__fo = ind
        self.l1_first: float = None
        self.l1_last: float = None
        self.jacobi: ndarray = array([0., 0.])

    def l1_minimum(self, x0: ndarray, jac: bool = False) -> float:
        """
        returns the cost function for a regression on the L2 norm
        l2 = sum( (f_i(measured) - f_i(calculated) ) ** 2 )
        :param x0: ndarray - [f0, b] such that
            f = i * x0[0] * sqrt(1. + x0[1] * i**2)
        :param jac: bool - if jacobi is to be calculated
        :return: float - l2 cost function
        other property:
        self.jacobi - ndarray - derivatives of dl1/df_0 and dl1/dB
        """
        l1 = 0.  # l1 cost function
        self.jacobi = array([0., 0.])
        # loop over peaks found
        for found in self.__fo:
            f_calc = found[1] * x0[0] * sqrt(1. + x0[1] * found[1] ** 2)
            diff = f_calc - found[0]
            l1 += diff * diff / found[0]  # L1 norm normalized to the frequency, as L1 increases from bass to discant
            if jac:
                # n-th partial is index + 1
                self.jacobi += self.__derivative(x0=x0, i=found[1], trova=found)

        if self.l1_first is None:
            self.l1_first = l1
        self.l1_last = l1

        return l1

    def l1_minimum_log_b(self, x0: ndarray, jac: bool = False) -> float:
        # b is coming in as log10(b)
        x0[1] = 10 ** x0[1]
        return self.l1_minimum(x0, jac)

    def l1_minimum_jac_direct(self, x0: ndarray) -> ndarray:
        self.l1_minimum(x0, jac=True)
        return self.jacobi

    def l1_minimum_jac(self, x0: ndarray) -> ndarray:
        return Jacobian(lambda x0: self.l1_minimum(x0))(x0).ravel()

    def l1_minimum_hess(self, x0: ndarray) -> ndarray:
        return Hessian(lambda x0: self.l1_minimum(x0))(x0)

    def l1_minimum_der(self, x0: ndarray) -> Tuple[float, ndarray]:
        return self.l1_minimum(x0, jac=True), self.jacobi

    def compare_l1(self) -> bool:
        return self.l1_last < self.l1_first

    @staticmethod
    def __derivative(x0: ndarray, i: int, trova: float) -> ndarray:
        if x0[1] < 0.:
            x0[1] = 0.
        tmp = sqrt(1. + x0[1] * i ** 2)
        tmp1 = (i * x0[0] * tmp - trova) / trova
        deriv_f0 = 2. * i * tmp * tmp1
        deriv_b = i ** 3 * x0[0] / tmp * tmp1

        return array([deriv_f0, deriv_b])
