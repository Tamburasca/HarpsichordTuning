from numpy import log10, ndarray
import logging
from scipy.optimize import brute, fmin
from typing import List, Tuple, TypeVar, Sequence
# internal
from Tuning.FFTaux import mytimer
from Tuning.L1costfunction import L1
from Tuning import parameters

_T = TypeVar('_T', bound=Sequence)

"""
note: these little auxiliary tools go with various minimizers. They
are disabled until needed and subject to being modified accordingly.


def bounds(f0, b):
    return (.996 * f0, 1.004 * f0), (max(0, .1 * b), min(2. * b, parameters.INHARM))


def constraints(f0, b):
    # since some methods do not accept boundaries,
    # we convert them to constraints, harhar
    bnds = asarray(bounds(f0, b))
    constraint = []
    for index in range(len(bnds)):
        lower, upper = bnds[index]
        # lower constraint first, upper second
        constraint.append(
            {'type': 'ineq', 'fun': lambda x, lb=lower, i=index: x[i] - lb})
        constraint.append(
            {'type': 'ineq', 'fun': lambda x, ub=upper, i=index: ub - x[i]})
    return constraint


class MyBounds(object):
    def __init__(self, f0):
        self.xmax = array([1.005 * f0, parameters.INHARM])
        self.xmin = array([0.995 * f0, 0.])

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(all(x <= self.xmax))
        tmin = bool(all(x >= self.xmin))
        return tmax and tmin


class MyTakeStep:
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize
        self.rng = random.default_rng()

    def __call__(self, x):
        s = self.stepsize
        x[0] += self.rng.uniform(-1. * s, 1. * s)
        x[1] += self.rng.uniform(-1.e-5 * s, 1.e-5 * s)
        return x


def myslice(x0: List) -> Tuple[_T, _T]:
    f0 = x0[0]
    b = max(0., x0[1])
    return slice(0.996 * f0,
                 1.0041 * f0,
                 0.002 * f0), slice(0.1 * b,
                                    min(2.11 * b, parameters.INHARM),
                                    0.5 * b)
"""


def myslicelog(x0: List) -> Tuple[_T, _T]:
    f0 = x0[0]
    b = max(1.e-7, x0[1])
    return slice(0.993 * f0,
                 1.0071 * f0,
                 0.002 * f0), \
           slice(log10(b) - 1.0,
                 min(log10(b) + .51, log10(parameters.INHARM)),
                 0.25)


@mytimer("L1 Minimization")
def final_fit(av: ndarray, ind: List[Tuple]) -> Tuple[float, float]:
    """
    fits the base frequency, inharmonicity by minimizing the L1 cost function
    as the deviation from the measured resonance frequencies to the
    calculated frequencies f = i * res.x0[0] * sqrt(1. + res.x0[1] * i**2),
    where i is the partial.
    We note that the minimizing of the L1 norm is achieved by converting the
    inharmonicity b into its log10 before being called, such that the grid
    for the brute force minimizer is equidistant in log space.
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
    guess = [av[5], log10(av[4])]
    l1_min = L1(ind)
    l1_min.l1_minimum_log_b(guess)
    logging.debug("Brute force grids: f0={0}, B={1}".format(myslicelog(guess)[0], myslicelog(guess)[1]))
    try:
        # do NOT use workers, muliprocessing's overhead slows it down
        res = brute(func=l1_min.l1_minimum_log_b,
                    ranges=myslicelog(guess),
                    finish=fmin,
                    full_output=True,
                    )

        def debug_msg(success: bool) -> None:
            logging.debug("Minimizer: Success: {0} L1 initial value: {1}, last value: {2}\n\t"
                          "fit result: base freq. = {3:.4f} Hz, B = {4:.3e}".format(
                              success,
                              l1_min.l1_first, res[1],
                              res[0][0], 10 ** res[0][1]))

        if res[1] <= l1_min.l1_first:
            debug_msg(True)
            return res[0][0], 10**res[0][1]
        else:
            # if L1 min final is worse than that with its initial values,
            # resume with the unchanged values
            debug_msg(False)
            return av[5], av[4]
    except Exception as e:
        logging.warning(str(e))
        return av[5], av[4]
