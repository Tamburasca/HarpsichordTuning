import logging
from scipy.optimize import minimize
from typing import List, Tuple, Sequence
from numpy.typing import NDArray
# internal
from FFTaux import mytimer
from L1costfunction import L1
# from Tuning.FFTL2minimizer import L2
import parameters


def callback(xk: NDArray) -> bool:
    # toggle for minimizer analysis -> L1_contours
    # print("[{},{}],".format(xk[0], xk[1]))

    return False


def bounds(x0: NDArray) -> Sequence:
    f0 = x0[0]
    b = max(0., x0[1])

    return (.995 * f0, 1.005 * f0), (.05 * b, min(5. * b, parameters.INHARM))


@mytimer("L1 Minimization")
def final_fit(
        av: NDArray,
        ind: List
) -> Tuple[float, float]:
    """
    fits the base frequency and inharmonicity by minimizing the L1 cost function
    as the deviation from the measured resonance frequencies to the
    calculated frequencies f = i * res.x0[0] * sqrt(1. + res.x0[1] * i**2),
    where i is the partial
    :param av: array - [lower, upper partials, lower, upper frequencies,
    inharmonicity, and base frequency]
    :param ind: array - measured resonance frequencies as from FFT
    :return: float, float - base frequency, inharmonicity (if success: fit
    result, else returns input values
    note:
    https://stackoverflow.com/questions/41137092/jacobian-and-hessian-inputs-in-scipy-optimize-minimize
    """
    if av[4] <= 0:
        return av[5], av[4]
    guess = [av[5], av[4]]
    l1_min = L1(ind)
    l1_min.l1_minimum(x0=guess)
    try:
        '''
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
        '''
        res = minimize(fun=l1_min.l1_minimum_der,
                       x0=guess,
                       bounds=bounds(guess),
                       method='SLSQP',
                       jac=True,
                       callback=callback,
                       options={})

        def debug_msg(success: bool) -> None:
            logging.debug("Minimizer: Success: {0} L1 initial value: {1}, "
                          "last value: {2}\n\t"
                          "number of iterations/evaluation: {3}/{4}\n\t"
                          "message: {5}".format(
                                success,
                                l1_min.l1_first, res.fun,
                                res.nit, res.nfev,
                                res.message))

        if l1_min.l1_first > res.fun:
            debug_msg(True)
            return res.x[0], res.x[1]
        else:
            debug_msg(False)
            return av[5], av[4]

    except Exception as e:
        logging.warning(str(e))
        return av[5], av[4]
