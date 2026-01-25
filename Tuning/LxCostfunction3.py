"""
cost function of the L1 and L2 norm, to be minimized.

L1 being the sum of the absolute
deviations of the calculated resonance frequencies from the measured ones,
normalized to the measured frequencies.

L2 being the sum of the squared
deviations of the calculated resonance frequencies from the measured ones,
normalized to the squared measured frequencies.

Jacobian and Hessian can be calculated either analytically or with finite
difference approximation.
"""

from numdifftools import Jacobian, Hessian
from numpy import sqrt, array, sign, nan, zeros
from numpy.typing import NDArray

from parameters import FREQUENCY_LOWER

I_MAX = int(16_000 / FREQUENCY_LOWER)


class L1(object):
    """
    cost function of the L1 norm, to be minimized.
    """

    def __init__(
            self,
            ind: list
    ) -> None:
        """
        :param ind: list of tuples -
        1st element of tuple: measured resonance frequencies as from peaks (FFT)
        2nd element of tuple: its partial as determined by peak selection
        """
        self.__fo = ind
        self.lx_first: float = nan
        self.lx_last: float = nan
        self.jacobi: NDArray = zeros(2, dtype=float)
        self.jacobi.fill(nan)

    def lx_minimum(
            self,
            x0: NDArray,
            jac: bool = False
    ) -> float:
        """
        returns the cost function for a regression on the L1 norm
        l1 = sum( abs( f_i(measured) - f_i(calculated) ) / f_i(measured) )
        :param x0 - [f0, b] provided by minimizer, such that
            f = i * x0[0] * sqrt(1. + x0[1] * i**2)
            inharmonicity of a vibrating string for partial i
        :param jac - if jacobi is to be calculated
        :return - l1 cost function
        other property:
        if jac is True the Jacobian is calculated analytically and stored in
        self.jacobi - derivatives δl1/δf_0 and δl1/δB
        """
        f0, b = x0[0], max(x0[1], 0.)
        l1 = 0.  # l1 cost function
        if jac: self.jacobi = array([0., 0.])

        j = 1
        # loop over peaks found
        for found in self.__fo:
            fl = j * f0 * sqrt(1. + b * j ** 2)

            for i in range(j, I_MAX):  # loop over partials until found match
                fh = (i + 1) * f0 * sqrt(1. + b * (i + 1) ** 2)
                j = i

                if found < fl and i == 1:
                    diff = fl - found
                    partial = 1
                elif fl <= found < fh:
                    if (found - fl) < (fh - found):
                        diff = fl - found
                        partial = i
                    else:
                        diff = fh - found
                        partial = i + 1
                else:
                    fl = fh
                    continue

                l1 += abs(diff) / found
                if jac:
                    self.jacobi += self.__derivative(
                        x0=array([f0, b]),
                        i=partial,
                        trova=found
                    ) * sign(diff)
                break

        if self.lx_first is nan: self.lx_first = l1
        self.lx_last = l1

        return l1

    def lx_minimum_log_b(
            self,
            x0: NDArray,
            jac: bool = True
    ) -> tuple[float, NDArray]:
        """
        b is coming in as log10(b), used with bruteforce. Not used with SLSQP
        :param x0:
        :param jac:
        :return:
        """
        return self.lx_minimum(array([x0[0], 10 ** x0[1]]), jac), self.jacobi

    def lx_minimum_jac_direct(self, x0: NDArray) -> NDArray:
        """
        not used, but kept for reference
        :param x0:
        :return:
        """
        self.lx_minimum(x0, jac=True)
        return self.jacobi

    def lx_minimum_jac(self, x0: NDArray) -> NDArray:
        """
        Calculate Jacobian with finite difference approximation
        :param x0:
        :return:
        """
        return Jacobian(self.lx_minimum(x0))(x0).ravel()

    def lx_minimum_hess(self, x0: NDArray) -> NDArray:
        """
        Calculate Hessian with finite difference approximation
        :param x0:
        :return:
        """
        return Hessian(self.lx_minimum(x0))(x0)

    def lx_minimum_der(self, x0: NDArray) -> tuple[float, NDArray]:
        """
        returns both the L1 cost function and its derivatives, computed analytically
        :param x0:
        :return:
        """
        return self.lx_minimum(x0, jac=True), self.jacobi

    def compare_lx(self) -> bool:
        """compare first and last L1 cost function calculation."""
        return self.lx_last < self.lx_first

    @staticmethod
    def __derivative(
            x0: NDArray,
            i: int,
            trova: float
    ) -> NDArray:
        """
        computes analytically the normalized derivatives of the L1 cost
        function with respect to f0 and b for an individual partial i
        :param x0:
        :param i: partial number
        :param trova: measured frequency at partial i
        :return:
        """
        tmp = sqrt(1. + x0[1] * i ** 2)

        # derivative with respect to base frequency
        deriv_f0 = i * tmp / trova
        # derivative with respect to inharmonicity
        deriv_b = 0.5 * i ** 3 * x0[0] / tmp / trova

        return array([deriv_f0, deriv_b])


class L2(object):
    """
    cost function of the L2 norm, to be minimized.
    """

    def __init__(
            self,
            ind: list
    ) -> None:
        """
        :param ind: list of tuples -
        1st element of tuple: measured resonance frequencies as from peaks (FFT)
        2nd element of tuple: its partial as determined by peak selection
        """
        self.__fo = ind
        self.lx_first: float = nan
        self.lx_last: float = nan
        self.jacobi: NDArray = zeros(2, dtype=float)
        self.jacobi.fill(nan)

    def lx_minimum(
            self,
            x0: NDArray,
            jac: bool = False
    ) -> float:
        """
        returns the cost function for a regression on the L2 norm
        l2 = sum( ((f_i(measured) - f_i(calculated) ) / f_i(measured)) ** 2 )
        :param x0: NDArray - [f0, b] such that
            f = i * x0[0] * sqrt(1. + x0[1] * i**2)
            inharmonicity of a vibrating string for partial i
        :param jac - if jacobi is to be calculated
        :return - l2 cost function
        other property:
        if jac is True the Jacobian is calculated analytically and stored in
        self.jacobi - derivatives δl2/δf_0 and δl2/δB
        """
        f0, b = x0[0], max(x0[1], 0.)
        l2 = 0.  # l2 cost function
        if jac: self.jacobi = array([0., 0.])

        j = 1
        # loop over peaks found
        for found in self.__fo:
            fl = j * f0 * sqrt(1. + b * j ** 2)

            for i in range(j, I_MAX):  # loop over partials until found match
                fh = (i + 1) * f0 * sqrt(1. + b * (i + 1) ** 2)
                j = i

                if found < fl and i == 1:
                    diff = fl - found
                    partial = 1
                elif fl <= found < fh:
                    if (found - fl) < (fh - found):
                        diff = fl - found
                        partial = i
                    else:
                        diff = fh - found
                        partial = i + 1
                else:
                    fl = fh
                    continue

                l2 += diff * diff / found / found
                if jac:
                    self.jacobi += self.__derivative(
                        x0=array([f0, b]),
                        i=partial,
                        trova=found
                    )
                break

        if self.lx_first is nan: self.lx_first = l2
        self.lx_last = l2

        return l2

    def lx_minimum_log_b(
            self,
            x0: NDArray,
            jac: bool = True
    ) -> tuple[float, NDArray]:
        """
        b is coming in as log10(b), used with bruteforce. Not used with SLSQP
        :param x0:
        :param jac:
        :return:
        """
        return self.lx_minimum(array([x0[0], 10 ** x0[1]]), jac), self.jacobi

    def lx_minimum_jac_direct(self, x0: NDArray) -> NDArray:
        """
        not used, but kept for reference
        :param x0:
        :return:
        """
        self.lx_minimum(x0, jac=True)
        return self.jacobi

    def lx_minimum_jac(self, x0: NDArray) -> NDArray:
        """
        Calculate Jacobian with finite difference approximation
        :param x0:
        :return:
        """
        return Jacobian(self.lx_minimum(x0))(x0).ravel()

    def lx_minimum_hess(self, x0: NDArray) -> NDArray:
        """
        Calculate Hessian with finite difference approximation
        :param x0:
        :return:
        """
        return Hessian(self.lx_minimum(x0))(x0)

    def lx_minimum_der(self, x0: NDArray) -> tuple[float, NDArray]:
        """
        returns both the L2 cost function and its derivatives, computed analytically
        :param x0:
        :return:
        """
        return self.lx_minimum(x0, jac=True), self.jacobi

    def compare_lx(self) -> bool:
        """compare first and last L2 cost function calculation."""
        return self.lx_last < self.lx_first

    @staticmethod
    def __derivative(
            x0: NDArray,
            i: int,
            trova: float
    ) -> NDArray:
        """
        computes analytically the normalized derivatives of the L2 cost
        function with respect to f0 and b for an individual partial i
        :param x0:
        :param i: partial number
        :param trova: measured frequency at partial i
        :return:
        """
        tmp = sqrt(1. + x0[1] * i ** 2)
        tmp1 = (i * x0[0] * tmp - trova) / trova / trova

        # derivative with respect to base frequency
        deriv_f0 = 2. * i * tmp * tmp1
        # derivative with respect to inharmonicity
        deriv_b = i ** 3 * x0[0] / tmp * tmp1

        return array([deriv_f0, deriv_b])
