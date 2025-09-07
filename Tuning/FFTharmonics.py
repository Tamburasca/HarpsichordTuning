import logging
from math import gcd
from operator import itemgetter
from typing import List, Tuple, Dict

from numpy import sqrt, append, mean, array

import parameters
# internal
from FFTaux import mytimer
# from Tuning.minimize_bruteforce import final_fit
from minimize_SLSQP import final_fit


def bisection(
        vector: List,
        value: float
) -> int:
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


def l1min(
        ind: List,
        x0: List
) -> float:
    """
    returns the cost function for a regression on the L1 norm
    l1 = sum( abs( f_i(measured) - f_i(calculated) ) / f_i(measured) )
    :param ind: list - measured resonance frequencies as from peaks (FFT)
    after being cleansed, dublicates removed, etc.
    :param x0: list - [f0, b] such that f = i * x0[0] * sqrt(1. + x0[1] * i**2)
    :return: float - l1 cost function
    """
    fmax = max(ind)
    l1 = 0.  # l1 cost function
    freq = list()
    for i in range(1, 640):
        f = i * x0[0] * sqrt(1. + x0[1] * i ** 2)
        freq.append(f)
        if f > fmax:
            break  # exit if superseded max. frequency measured to save time
    num_freq_calc = len(freq)
    # loop over peaks found
    for found in ind:
        idx = bisection(freq, found)
        if idx == -1:
            # <min frequency
            ids = 0
        elif idx == num_freq_calc:
            # >max frequency
            ids = num_freq_calc - 1
        else:
            # consider the closest candidate of neighbors
            ids = idx \
                if abs(found - freq[idx]) < abs(found - freq[idx + 1]) \
                else idx + 1
        diff = freq[ids] - found
        # L1 norm normalized to the frequency, as L1 increases from bass to discant
        l1 += abs(diff) / found

    return l1


def select_list(selected: List[List]) -> List[Tuple]:
    """
    list of resonance peaks according to harmonics - remove dublettes with same
    upper frequency tagged with upper partial
    :param selected: list of selected peaks
    :return: list of tuples (resonance peaks, upper partial)
    """
    identified = dict({(selected[0][2],): selected[0][0]})
    for item in selected:
        identified[(item[3],)] = item[1]
    # toggle for minimizer analysis -> L1_contours
    # print([(key, int(value)) for (key,), value in identified.items()])
    return [(key, int(value)) for (key,), value in identified.items()]


@mytimer("harmonics (subtract time for L1 minimization, if called)")
def harmonics(peaks: List[Tuple]) -> List:
    """
    finds harmonics between each two frequencies by applying the inharmonicity
    formula by a neested loop through all the peaks
    :param peaks: list
        tuples of freqencies and amplitudes of FFT transformed spectrum
    :return:
    list (float)
        positions of first NPARTIAL partials
    """
    initial: List = []
    l1: Dict[int, List] = {}
    f_n: List = []

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
                        logging.info(
                            "devideByZero: discarded value in harmonics finding"
                        )
                        continue
                    if -0.0001 < b < parameters.INHARM:
                        # allow also negative b value > -0.0001 for
                        # uncertainties in the line fitting
                        f_fundamental = ind[i] / (m * sqrt(1. + b * m ** 2))
                        if (parameters.FREQUENCY_LOWER
                                > f_fundamental
                                > parameters.FREQUENCY_UPPER):
                            break  # break two loops here
                        element = [
                            m, k, ind[i], ind[j], max(b, 0.), f_fundamental
                        ]  # always b >= 0
                        if initial:
                            if (element[3] == initial[-1][3]
                                    and element[0] == initial[-1][0]):
                                # remove previous dublette on upper frequency and lower partial
                                initial.pop()
                        initial.append(element)
                        break  # break two loops here
                break

    for item in initial:
        # prepare for two partials with no common divisor open an empty list
        if gcd(item[0], item[1]) == 1:
            if item[0] not in l1:
                # create dict keys with empty list values
                l1[item[0]] = list()
        logging.debug(
            "partials: {0:2d} {1:2d} lower: {2:10.4f} upper: {3:10.4f} "
            "B: {4: .1e} fundamental: {5:10.4f}".format(*item))

    """
    disregard fundamentals for records, where both partials have a common 
    divisor (gcd). Consider all fundamentals with no common divisor only.
    """
    if initial:
        av = array([])
        selected = array([])
        no_of_peak_combi = 0

        if len(l1) > 1:
            # if more than one lower partials with gcd=1
            for key in l1:
                for dat in filter(lambda x: x[0] == key, initial):
                    # Add all l1 values to list for same lower partial
                    l1[key].append(l1min(ind=ind, x0=[dat[5], dat[4]]))
                # l1 cost function averaged for equal lower partials
                l1[key] = mean(l1[key])
            # identify lower partial with minimum l1
            selected = array(
                list(
                    filter(lambda x: x[0] == min(l1, key=l1.get), initial))
            )
            av = selected.mean(axis=0)
            no_of_peak_combi = selected.shape[0]
            logging.debug("L1: {}".format(l1))
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
        if (parameters.FREQUENCY_LOWER
                < base_frequency
                < parameters.FREQUENCY_UPPER):
            if no_of_peak_combi > 1:
                identified = select_list(selected=selected)
                base_frequency, inharmonicity = final_fit(
                    av=av,
                    ind=identified
                )
                logging.debug(
                    "Initial: f_0 = {0:.3f} Hz, B = {1:.3e} "
                    "Final: f_0 = {2:.3f} Hz, B = {3:.3e}".format(
                        av[5], av[4], base_frequency, inharmonicity)
                )
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
        if parameters.FREQUENCY_LOWER < f1 < parameters.FREQUENCY_UPPER:
            f_n.append(f1)
            logging.info(
                "Best result: f_1 = {0:.2f} Hz, B = {1:.1e}".format(f1, 0.)
            )

    return f_n
