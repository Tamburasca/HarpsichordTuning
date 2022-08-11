from numpy import sqrt, append, mean, array
from math import gcd
import logging
from operator import itemgetter
from scipy.optimize import minimize

from Tuning.FFTaux import mytimer
from Tuning.FFTL1minimizer import L1
# from Tuning.FFTL2minimizer import L2
from Tuning import parameters as P


def callback(xk):
    print("[{},{}],".format(xk[0], xk[1]))


def bounds(x0):
    f0 = x0[0]
    b = max(0., x0[1])
    return (.998 * f0, 1.002 * f0), \
           (.05 * b, min(3. * b, P.INHARM))


@mytimer("L1 Minimization")
def final_fit(av, ind):
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
        res = minimize(fun=l1_min.l1_minimum_der,
                       x0=guess,
                       bounds=bounds(guess),
                       method='SLSQP',
                       jac=True,
                       # callback=callback,
                       options={})
        # print(res)
        msg = "Minimizer: Success: {0} L1 initial value: {1}, last value: {2}\n\t" \
              "number of iterations/evaluation: {3}/{4}"
        if res.success:
            if l1_min.l1_first > res.fun:
                logging.debug(msg.format(
                        True,
                        l1_min.l1_first, res.fun,
                        res.nit, res.nfev))
                return res.x[0], res.x[1]
            else:
                logging.debug(msg.format(
                        False,
                        l1_min.l1_first, res.fun,
                        res.nit, res.nfev))
                return av[5], av[4]
        else:
            if l1_min.l1_first > res.fun:
                logging.warning(msg.format(
                        True,
                        l1_min.l1_first, res.fun,
                        res.nit, res.nfev))
                return res.x[0], res.x[1]
            else:
                logging.warning(msg.format(
                        False,
                        l1_min.l1_first, res.fun,
                        res.nit, res.nfev))
                return av[5], av[4]
    except Exception as e:
        logging.warning(str(e))
        return av[5], av[4]


@mytimer("harmonics (subtract time for L1 minimization, if called)")
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
        no_of_peak_combi = 0

        if len(l1) > 1:
            # if more than one lower partials with gcd=1
            for key in l1:
                for dat in filter(lambda x: x[0] == key, initial):
                    # Add all l1 values to list for same lower partial
                    l1[key].append(l1_min.l1_minimum(x0=[dat[5], dat[4]]))
                # l1 cost function averaged for equal lower partials
                l1[key] = mean(l1[key])
            # identify lower partial with minimum l1
            selected = array(
                list(
                    filter(lambda x: x[0] == min(l1, key=l1.get), initial))
            )
            av = selected.mean(axis=0)
            no_of_peak_combi = selected.shape[0]

        elif len(l1) == 1:
            # if only one lower partial with gcd=1
            selected = array(
                list(
                    filter(lambda x: x[0] == list(l1.keys())[0], initial))
            )
            av = selected.mean(axis=0)
            no_of_peak_combi = selected.shape[0]

        if av.size == 0:
            # if no gcd=1 found, take first entry for the lowest partial
            av = array(list(
                filter(lambda x: x[0] == initial[0][0], initial))
            ).mean(axis=0)

        base_frequency = av[5]
        inharmonicity = av[4]
        if base_frequency > P.FREQUENCY_LIMIT:
            if no_of_peak_combi > 1 and P.FINAL_FIT:
                base_frequency, inharmonicity = final_fit(av=av, ind=ind)
                logging.debug("Initial: f_0 = {0:.3f} Hz, B = {1:.3e} "
                              "Final: f_0 = {2:.3f} Hz, B = {3:.3e}".format(
                                  av[5], av[4], base_frequency, inharmonicity)
                              )
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
