import logging
from collections.abc import Sequence
from typing import Callable

from numpy import array
from numpy.typing import NDArray
from scipy.optimize import minimize, OptimizeResult

# internal
import parameters
from FFTaux import mytimer
from LxCostfunction import L1, L2


def callback(xk: NDArray) -> None:
    # toggle for minimizer analysis -> Lx_contours
    # print("[{},{}],".format(xk[0], xk[1]))
    pass


def bounds(x0: NDArray) -> Sequence[tuple[float, float]]:
    """
    bounds for base frequency and inharmonicity
    :param x0: frequency and inharmonicity initial guess
    :return:
    """
    f0 = x0[0]
    b = max(0., x0[1])

    return (.995 * f0, 1.005 * f0), (.05 * b, min(5. * b, parameters.INHARM))


def minimizer(
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
        bounds=bounds(x0),
        method='SLSQP',
        jac=True,
        callback=callback,
        options=None
    )


@mytimer(f"{parameters.COST_FUNCTION}-Minimization")
def final_fit(
        av: NDArray,
        ind: list[tuple[float, int]]
) -> tuple[float, float]:
    """
    fits the base frequency and inharmonicity by minimizing the Lx cost function
    as the deviation from the measured resonance frequencies to the
    calculated frequencies f = i * res.x0[0] * sqrt(1. + res.x0[1] * i**2),
    where i is the partial
    :param av: array - [lower, upper partials, lower, upper frequencies,
    inharmonicity, and base frequency]
    :param ind: array - measured resonance frequencies as from FFT
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
    if av[4] <= 0:
        return av[5], av[4]

    x0 = array([av[5], av[4]])
    try:
        if parameters.COST_FUNCTION == 'L1':
            l1_min = L1(ind)
            l1_min.l1_minimum(x0=x0)
            res = minimizer(fun=l1_min.l1_minimum_der, x0=x0)
            l_first = l1_min.l1_first
        else:  # L2
            l2_min = L2(ind)
            l2_min.l2_minimum(x0=x0)
            res = minimizer(fun=l2_min.l2_minimum_der, x0=x0)
            l_first = l2_min.l2_first

        if l_first > res.fun:
            logging.debug(
                f"{parameters.COST_FUNCTION}-Minimizer: Success: True\n\t"
                f"initial value: {l_first}, last value: {res.fun}\n\t"
                f"number of iterations/evaluation: {res.nit}/{res.nfev}\n\t"
                f"message: {res.message}")
            return res.x
        else:
            logging.debug(
                f"{parameters.COST_FUNCTION}-Minimizer: Success: False\n\t"
                f"initial value: {l_first}, last value: {res.fun}\n\t"
                f"number of iterations/evaluation: {res.nit}/{res.nfev}\n\t"
                f"message: {res.message}")
            return av[5], av[4]

    except Exception as e:
        logging.warning(str(e))
        return av[5], av[4]
