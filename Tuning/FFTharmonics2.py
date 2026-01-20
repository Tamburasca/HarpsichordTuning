import logging
import math
from math import isnan
from operator import itemgetter

from numpy import sqrt, mean, append, array, nan_to_num, gcd
from numpy.typing import NDArray

# internal
import parameters
from FFTaux import mytimer
from LxCostfunction2 import L1, L2
from minimize_SLSQP_class import MinimizeSLSQP

I_MAX = int(16_000 / parameters.FREQUENCY_LOWER)


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
    l1: dict[tuple[int, float], list[float]] = dict()
    f_n = list()

    # sort by frequency asc. and make list of indices (positions) and heights
    peaks.sort(key=lambda x: x[0])
    ind = list(map(itemgetter(0), peaks))
    height = list(map(itemgetter(1), peaks))
    logging.debug("ind: " + str(ind))
    logging.debug("height: " + str(height))

    if parameters.COST_FUNCTION == 'L1':
        lx_min = L1(ind)
    else:
        lx_min = L2(ind)

    next_low_partial = 1
    # loop through all peaks found (ascending, nested loops)
    for i in range(0, len(ind) - 1):  # lower freq.
        j = i + 1  # next upper freq. of neighboring peaks

        # loop through neighboring partials up to NPARTIAL
        for m in range(next_low_partial, parameters.NPARTIAL):  # lower partial
            for k in range(m + 1, parameters.NPARTIAL):  # upper partial
                # calculate inharmonicity factor b from two peaks ind[i], ind[j]
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
                    # calculate fundamental frequency from lower partial
                    f_fundamental = ind[i] / (m * sqrt(1. + b * m ** 2))
                    if not (parameters.FREQUENCY_LOWER
                            < f_fundamental
                            < parameters.FREQUENCY_UPPER):
                        break
                    element = [
                        m, k, ind[i], ind[j], max(b, 0.), f_fundamental
                    ]  # always b >= 0
                    if initial:
                        # remove if greatest common divisor >1 on each lower and
                        # upper when compared to last entry
                        if (gcd(element[0], initial[-1][0]) != 1
                                and gcd(element[1], initial[-1][1]) != 1):
                            break
                        # remove previous doublette on upper frequency and lower partial
                        if (element[3] == initial[-1][3]
                                and element[0] == initial[-1][0]):
                            # print("removed doublette", initial[-1], element)
                            initial.pop()
                    initial.append(element)

        next_low_partial += 1  # increase lower partial for next higher peak found

    if initial:
        l1_min = float('inf')

        for item in initial:  # if found any partial combinations
            initial_log = [
                # b: map zero to nan
                math.log(item[4]) if item[4] > 0.0 else float('nan'),
                item[5]
            ]
            # ToDo: key t may be obsolete!
            t = (item[0], item[2])  # combined key (lower part and lower freq.)
            if t not in l1:
                l1[t] = initial_log
            logging.debug(
                "partials: {0:2d} {1:2d} lower: {2:10.4f} upper: {3:10.4f} "
                "B: {4: .1e} fundamental: {5:10.4f}".format(*item))

        for _, val in l1.items():
            b_remapped = math.exp(val[0]) if not isnan(val[0]) else 0.
            t_new = lx_min.l1_minimum(x0=array([val[1], b_remapped]))
            if t_new < l1_min:  # choose if L1 is lower than previous
                l1_min = t_new
                logging.debug(
                    f"Last L1 minimum: {l1_min}, "
                    f"f0={float(val[1])}, "
                    f"b={b_remapped}")
                # initial guess of f0 and b for the Lx-minimizer
                base_frequency = val[1]
                inharmonicity = b_remapped

        if (parameters.FREQUENCY_LOWER
                < base_frequency
                < parameters.FREQUENCY_UPPER):
            base_frequency_final, inharmonicity_final = (
                MinimizeSLSQP(norm=parameters.COST_FUNCTION)(
                    ind=ind,
                    f0=base_frequency,
                    b=inharmonicity
                ))
            logging.debug(
                "initial: f_0 = {0:.3f} Hz, B = {1:.3e} "
                "Final: f_0 = {2:.3f} Hz, B = {3:.3e}".format(
                    base_frequency, inharmonicity,
                    base_frequency_final, inharmonicity_final)
            )
            # display synthetic spectrum
            for n in range(1, parameters.NPARTIAL):
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
                "Best result from strongest line: f_1 = {0:.2f} Hz, B = {1:.1e}"
                .format(f1, 0.)
            )

    return f_n
