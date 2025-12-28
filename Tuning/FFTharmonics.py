import logging
from math import gcd
from operator import itemgetter
from typing import Any

from numpy import sqrt, mean, append, array
from numpy.typing import NDArray

# internal
import parameters
from FFTaux import mytimer
# from minimize_bruteforce import final_fit
from minimize_SLSQP import final_fit

I_MAX = int(16_000 / parameters.FREQUENCY_LOWER)


def l1min_new(
        ind: list,
        x0: list
) -> float:
    """
    returns the cost function for a regression on the L1 norm
    l1 = sum( abs( f_i(measured) - f_i(calculated) ) / f_i(measured) )
    where f_i(measured) is considered to its nearest neighbor either
    f_i(calculated) or f_i+1(calculated)
    :param ind: list - measured resonance frequencies as from peaks (FFT)
    after being cleansed, duplicates removed, etc.
    :param x0: list - [f0, b] such that f = i * x0[0] * sqrt(1. + x0[1] * i**2)
    :return: float - l1 cost function
    """
    l1 = 0.  # l1 cost function
    j = 1
    # loop over peaks found
    for found in ind:
        fl = j * x0[0] * sqrt(1. + x0[1] * j ** 2)
        for i in range(j, I_MAX):
            fh = (i + 1) * x0[0] * sqrt(1. + x0[1] * (i + 1) ** 2)
            j = i
            if found < fl and i == 1:
                l1 += (fl - found) / found
                break
            elif fl <= found < fh:
                if (found - fl) < (fh - found):
                    diff = found - fl
                else:
                    diff = fh - found
                l1 += diff / found
                break
            else:
                fl = fh

    return l1


def select_list(selected: NDArray) -> list[tuple[float, int]]:
    """
    list of resonance peaks according to harmonics - remove doublettes with same
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


@mytimer(f"harmonics (minus time for {parameters.COST_FUNCTION} minimization)")
def harmonics(peaks: list[tuple]) -> list:
    """
    finds harmonics between each two frequencies by applying the inharmonicity
    formula by a nested loop through all the peaks
    :param peaks: list
        tuples of frequencies and amplitudes of FFT transformed spectrum
    :return:
    list (float)
        positions of first NPARTIAL partials
    """
    initial = list()
    l1: dict[int, list[float]] = dict()
    l1_mean: dict[int, Any] = dict()
    f_n = list()

    # sort by frequency asc. and make list of indices (positions) and heights
    peaks.sort(key=lambda x: x[0])
    ind = list(map(itemgetter(0), peaks))
    height = list(map(itemgetter(1), peaks))
    logging.debug("ind: " + str(ind))
    logging.debug("height: " + str(height))

    # loop through all combinations of partials up to NPARTIAL
    for m in range(1, parameters.NPARTIAL):
        for k in range(m + 1, parameters.NPARTIAL):
            # loop through all peaks found (ascending, nested loops)
            for i in range(0, len(ind)):
                for j in range(i + 1, len(ind)):
                    tmp = ((ind[j] * m) / (ind[i] * k)) ** 2
                    try:
                        b = (tmp - 1.) / (k ** 2 - tmp * m ** 2)
                    except ZeroDivisionError:
                        logging.info(
                            "divideByZero: discarded value in harmonics finding"
                        )
                        continue
                    if -0.0001 < b < parameters.INHARM:
                        # allow also negative b value > -0.0001 for
                        # uncertainties in the line fitting
                        f_fundamental = ind[i] / (m * sqrt(1. + b * m ** 2))
                        if not (parameters.FREQUENCY_LOWER
                                < f_fundamental
                                < parameters.FREQUENCY_UPPER):
                            break  # break two loops here
                        element = [
                            m, k, ind[i], ind[j], max(b, 0.), f_fundamental
                        ]  # always b >= 0
                        if initial:
                            if (element[3] == initial[-1][3]
                                    and element[0] == initial[-1][0]):
                                # remove previous doublette on upper frequency and lower partial
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
                    t_new = l1min_new(ind=ind, x0=[dat[5], dat[4]])
                    l1[key].append(t_new)
                # l1 cost function averaged for equal lower partials
                l1_mean[key] = mean(l1[key])
            # identify lower partial with minimum l1
            selected = array(
                list(
                    filter(lambda x: x[0] == min(l1_mean, key=l1_mean.get), initial))
            )
            av = selected.mean(axis=0)
            no_of_peak_combi = selected.shape[0]
            logging.debug("L1: {}".format(l1_mean))
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
            for n in range(1, I_MAX):
                # for n in range(1, parameters.NPARTIAL):
                f_synth = base_frequency * n * sqrt(
                    1. + inharmonicity * n ** 2)
                if f_synth < 12_000:
                    f_n = append(f_n, f_synth)  # show < 12.000 Hz
                else:
                    break
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
