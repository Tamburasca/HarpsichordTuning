from numpy import sqrt, append, array, mean
from math import gcd
import logging
from operator import itemgetter
from scipy.optimize import minimize

from Tuning.FFTaux import mytimer, l1_minimum
from Tuning import parameters


@mytimer("Lasso Regression")
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
    """
    guess = [av[5], av[4]]
    boundaries = [
        (av[5] * .995, av[5] * 1.005),
        (0., parameters.INHARM)
    ]
    try:
        res = minimize(fun=l1_minimum,
                       x0=guess,
                       bounds=boundaries,
                       args=(*ind,),
                       method='Nelder-Mead'  # ToDo: this is prelim.
                       )
        logging.debug(
            "Minimizer: Success: {0}, number of iterations: {1}, "
            "fit result: base freq. = {2:.4f} Hz, B = {3:.3e}".format(
                res.success,
                res.nit,
                res.x[0],
                res.x[1]))
        if res.success:
            return res.x[0], res.x[1]
        else:
            return av[5], av[4]
    except Exception as e:
        logging.warning(str(e))
        return av[5], av[4]


@mytimer
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
    for m in range(1, parameters.NPARTIAL):
        for k in range(m + 1, parameters.NPARTIAL):
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
                    if -0.0001 < b < parameters.INHARM:
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
        if len(l1) > 1:
            # if more than one lower partials with gcd=1
            for key in l1:
                for dat in filter(lambda x: x[0] == key, initial):
                    # Add all l1 values to list for same lower partial
                    l1[key].append(l1_minimum([dat[5], dat[4]], *ind))
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
        if base_frequency > parameters.FREQUENCY_LIMIT:
            if parameters.FINAL_FIT:
                base_frequency, inharmonicity = final_fit(av, ind)
            for n in range(1, parameters.NPARTIAL):
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
        if f1 > parameters.FREQUENCY_LIMIT:
            f_n.append(f1)
            logging.info(
                "Best result: f_1 = {0:.2f} Hz, B = {1:.1e}".format(f1, 0.)
            )

    return f_n
