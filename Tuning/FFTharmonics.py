from numpy import sqrt, append, mean, array, asarray, random
from math import gcd
import logging
from operator import itemgetter
from scipy.optimize import minimize, brute, fmin

from Tuning.FFTaux import mytimer, L1
from Tuning import parameters as P

"""
note: these little auxiliary tools work with various minimizers. They
are disabled until needed:

def bounds(f0, b):
    return (.995 * f0, 1.005 * f0), (max(0, .1 * b), min(10 * b, P.INHARM))


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
        self.xmax = array([1.005 * f0, P.INHARM])
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
"""


def myslice(x0):
    # ToDo slice dimensions
    return slice(0.998 * x0[0], 1.003 * x0[0], 0.001 * x0[0]), \
           slice(0.8 * x0[1], 1.2 * x0[1], 0.1 * x0[1])


@mytimer("L1 Minimization")
def final_fit(av, ind):
    """
    fits the base frequency, inharmonicity by minimizing the L1 cost function
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
    guess = [av[5], av[4]]
    l1_min = L1(ind)
    l1_min.l1_minimum(guess)
    try:
        '''
        res = minimize(fun=l1_min.l1_minimum,
                       x0=guess,
                       bounds=bounds(f0=av[5], b=av[4]),
                       # constraints=constraints(f0=av[5], b=av[4]),
                       method='Nelder-Mead',  # ToDo. find a better one
                       options={'return_all': False}
                       # jac=True
                       # jac=l1_min.l1_minimum_jac,
                       # hess=l1_min.l1_minimum_hess
                       )
        '''
        if av[4] == 0:
            return av[5], av[4]
        # do NOT use workers, muliprocessing's oeverhead slows it down
        res = brute(func=l1_min.l1_minimum,
                    ranges=myslice(guess),
                    finish=fmin,
                    full_output=True)
        msg = "Minimizer: Success: {0} L1 initial value: {1}, last value: {2}"
        "fit result: base freq. = {3:.4f} Hz, B = {4:.3e}"
        if res[1] <= l1_min.l1_first:
            logging.debug(msg.format(
                    True,
                    l1_min.l1_first, res[1],
                    res[0][0], res[0][1]))
            return res[0][0], res[0][1]
        else:
            # if L1 min final is worse than that with its initial values,
            # resume with the unchanged values
            logging.warning(msg.format(
                    False,
                    l1_min.l1_first, res[1],
                    res[0][0], res[0][1]))
            return av[5], av[4]
    except Exception as e:
        logging.warning(str(e))
        return av[5], av[4]


@mytimer("harmonics (subtract time for L1 minimization)")
def harmonics(peaks):
    """
    finds harmonics between each two frequencies by applying the inharmonicity
    formula by a neested loop through all the peaks
    :param peaks: list
        tuples of freqencies and amplitudes of FFT transformed spectrum
    :return:
    list (float)
        positions of first NPARTIAL partials
    """
    initial = list()
    l1 = dict()
    f_n = list()

    # sort by frequency ascending
    peaks.sort(key=lambda x: x[0])
    ind = list(map(itemgetter(0), peaks))
    height = list(map(itemgetter(1), peaks))
    logging.debug("ind: " + str(ind))
    logging.debug("height: " + str(height))

    # loop through the combination of partials up to NPARTIAL
    for m in range(1, P.NPARTIAL):
        for k in range(m + 1, P.NPARTIAL):
            # loop through all peaks found (ascending, neested loops)
            for i in range(0, len(ind)):
                for j in range(i + 1, len(ind)):
                    tmp = ((ind[j] * m) / (ind[i] * k)) ** 2
                    try:
                        b = (tmp - 1.) / (k ** 2 - tmp * m ** 2)
                    except ZeroDivisionError:
                        logging.info("devideByZero: "
                                     "discarded value in harmonics finding")
                        continue
                    if -0.0001 < b < P.INHARM:
                        # allow also negative b value > -0.0001 for
                        # uncertainties in the line fitting
                        f_fundamental = ind[i] / (m * sqrt(1. + b * m ** 2))
                        logging.debug(
                            "partial: {0:2d} {1:2d} "
                            "lower: {2:10.4f} upper: {3:10.4f} "
                            "b: {4: .1e} fundamental: {5:10.4f}"
                            .format(m, k, ind[i], ind[j], b, f_fundamental))
                        # always b >= 0
                        initial.append(
                            [m, k, ind[i], ind[j], max(b, 0.), f_fundamental]
                        )
                        # two partials with no common divisor
                        if gcd(m, k) == 1:
                            if m not in l1:
                                # create dict keys with empty list values
                                l1[m] = list()
                        break
                break
    """
    disregard fundamentals for records, where both partials have a common 
    divisor (gcd). Consider all fundamentals with no common divisor only.
    """
    if initial:
        av = array([])
        l1_min = L1(ind)
        if len(l1) > 1:
            # if more than one lower partials with gcd=1
            for key in l1:
                for dat in filter(lambda x: x[0] == key, initial):
                    # Add all l1 values to list for same lower partial
                    l1[key].append(l1_min.l1_minimum([dat[5], dat[4]]))
                # l1 cost function averaged for equal lower partials
                l1[key] = mean(l1[key])
            # identify lower partial with min l1
            av = array(list(
                filter(lambda x: x[0] == min(l1, key=l1.get), initial))
            ).mean(axis=0)
        elif len(l1) == 1:
            # if only one lower partial with gcd=1
            av = array(list(
                filter(lambda x: x[0] == list(l1.keys())[0], initial))
            ).mean(axis=0)
        if av.size == 0:
            # if no gcd=1 found, take first entry for the lowest partial
            av = array(list(
                filter(lambda x: x[0] == initial[0][0], initial))
            ).mean(axis=0)
        base_frequency = av[5]
        inharmonicity = av[4]
        if base_frequency > P.FREQUENCY_LIMIT:
            if P.FINAL_FIT:
                base_frequency, inharmonicity = final_fit(av, ind)
            for n in range(1, P.NPARTIAL):
                f_n = append(f_n,
                             base_frequency * n * sqrt(
                                 1. + inharmonicity * n ** 2))
            logging.info(
                "Best result: f_1 = {0:.2f} Hz, B = {1:.1e}".format(
                    f_n[0], inharmonicity)
            )
    elif not initial and len(ind) > 0:
        # if fundamental could not be calculated through at least two lines,
        # give it a shot with the strongest peak found
        peaks.sort(key=lambda x: x[1], reverse=True)  # sort by amplitude desc
        f1 = list(map(itemgetter(0), peaks))[0]
        if f1 > P.FREQUENCY_LIMIT:
            f_n.append(f1)
            logging.info(
                "Best result: f_1 = {0:.2f} Hz, B = {1:.1e}".format(f1, 0.)
            )

    return f_n
