import logging
from collections.abc import Sequence
from typing import Callable

from numpy import array, append
from numpy.typing import NDArray
from scipy.optimize import minimize, OptimizeResult

# internal
import parameters
from FFTaux import mytimer
from LxCostfunction2 import L1, L2


class MinimizeSLSQP(object):
    def __init__(
            self,
            norm: str = 'L1'
    ) -> None:
        assert norm in ['L1', 'L2'], "MinimizeSLSQP: norm must be 'L1'|'L2'"
        if norm == 'L1':
            self.options = None  # {'ftol': 1.e-11}
        else:
            self.options = {'ftol': 1.e-10}
        self.norm = norm
        self.path: NDArray = array([], dtype=float).reshape(0, 2)
        self.success: bool = False

    def callback(
            self,
            xk: NDArray
    ) -> None:
        if self.norm == 'L1':
            self.path = append(self.path, [xk.copy()], axis=0)
        else:
            self.path = append(self.path, [xk.copy()], axis=0)

    @staticmethod
    def bounds(x0: NDArray) -> Sequence[tuple[float, float]]:
        """
        bounds for base frequency and inharmonicity
        :param x0: frequency and inharmonicity initial guess
        :return:
        """
        f0 = x0[0]
        b = max(0., x0[1])

        return (.995 * f0, 1.005 * f0), (.05 * b, min(10. * b, parameters.INHARM))

    def minimizer(
            self,
            fun: Callable,
            x0: NDArray
    ) -> OptimizeResult:
        """
        SLSQP minimizer
        :param fun: function to minimize
        :param x0: frequency and inharmonicity initial guess
        :return:
        """
        return minimize(
            fun=fun,
            x0=x0,
            bounds=MinimizeSLSQP.bounds(x0),
            method='SLSQP',
            jac=True,
            callback=self.callback,
            options=self.options
        )

    def msg(
            self,
            success: bool,
            l_first: float,
            res: OptimizeResult
    ) -> None:
        logging.debug(
            f"{self.norm}-Minimizer: Success: {success}\n\t"
            f"initial value: {l_first}, last value: {res.fun}\n\t"
            f"number of iterations/evaluation: {res.nit}/{res.nfev}\n\t"
            f"message: {res.message}")

    @mytimer("Lx-Minimization")
    def __call__(
            self,
            *,
            ind: list,
            f0: float,
            b: float
    ) -> tuple[float, float]:
        """
        fits the base frequency and inharmonicity through minimizing the Lx cost
        function as the deviation from the measured resonance frequencies to the
        calculated frequencies f = i * res.x0[0] * sqrt(1. + res.x0[1] * i**2),
        where i is the partial
        :param ind: array - measured resonance frequencies as from FFT
        :param f0: float - base frequency initial guess
        :param b: float - inharmonicity initial guess
        :return - base frequency, inharmonicity, if success: fit
        result, else returns input values
        note:
        https://stackoverflow.com/questions/41137092/jacobian-and-hessian-inputs-in-scipy-optimize-minimize

        bruteforce approach as is dismissed:
        res = minimize(fun=l1_min.l1_minimum_der,
                   x0=guess,
                   bounds=bounds(f0=av[5], b=av[4]),
                   # constraints=constraints(f0=av[5], b=av[4]),
                   method='BFGS',
                   options={'return_all': False},
                   jac=True
                   # jac=l1_min.l1_minimum_jac,
                   # hess=l1_min.l1_minimum_hess
                   )
        """
        assert b >= 0, "MinimizeSLSQP: inharmonicity must be >= 0"
        x0 = array([f0, b])
        try:
            if self.norm == 'L1':
                l1_min = L1(ind)
                l1_min.l1_minimum(x0=x0)
                l_first = l1_min.l1_first
                res = self.minimizer(fun=l1_min.l1_minimum_der, x0=x0)
            else:  # L2
                l2_min = L2(ind)
                l2_min.l2_minimum(x0=x0)
                l_first = l2_min.l2_first
                res = self.minimizer(fun=l2_min.l2_minimum_der, x0=x0)
        except Exception as e:
            logging.warning(str(e))
            return f0, b

        # toggle for optimize.minimize Lx analysis -> Lx contours
        # print(self.path)

        if l_first >= res.fun and res.success:
            self.msg(success=True, l_first=l_first, res=res)
            self.res = res
            self.success = True

            if self.norm == 'L1':
                return res.x
            else:
                return res.x

        else:
            self.msg(success=False, l_first=l_first, res=res)
            return f0, b
