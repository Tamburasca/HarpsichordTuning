import logging
import math
from math import isnan
from operator import itemgetter

from numpy import sqrt, append, array, nan

# internal
import parameters
from FFTaux import mytimer
from LxCostfunction3 import L1, L2
from minimize_SLSQP_class import MinimizeSLSQP

THRESHOLD = 16_000
I_MAX = int(THRESHOLD / parameters.FREQUENCY_LOWER)


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
    lx: dict[tuple[int, float], list[float]] = dict()
    f_n = list()
    base_frequency: float = nan
    inharmonicity: float = 0.

    # sort by frequency asc. and make list of indices (positions) and heights
    peaks.sort(key=lambda x: x[0])
    ind = list(map(itemgetter(0), peaks))
    height = list(map(itemgetter(1), peaks))
    logging.debug("ind: " + str(ind))
    logging.debug("height: " + str(height))

    if parameters.COST_FUNCTION == 'L1':
        lx_norm = L1(ind)
    else:
        lx_norm = L2(ind)

    next_low_partial = 1
    # loop through all peaks found (ascending)
    for i in range(0, len(ind) - 1):  # lower freq.
        j = i + 1  # next adjacent upper peak

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
                        # remove combinations with same lower & upper peak
                        # positions for all higher partials
                        if (element[2] == initial[-1][2]
                                and element[3] == initial[-1][3]):
                            break
                    initial.append(element)

        next_low_partial += 1  # increase lower partial for next higher peak found

    if initial:
        lx_norm.lx_first = float('inf')

        for item in initial:  # if found any partial combinations
            initial_log = [
                # b: map zero to nan
                math.log(item[4]) if item[4] > 0.0 else float('nan'),
                item[5]
            ]
            # ToDo: key t may be obsolete!
            t = (item[0], item[2])  # combined key (lower part and lower freq.)
            if t not in lx:
                lx[t] = initial_log
            logging.debug(
                "partials: {0:2d} {1:2d} lower: {2:10.4f} upper: {3:10.4f} "
                "B: {4: .1e} fundamental: {5:10.4f}".format(*item))

        for _, val in lx.items():
            b_remapped = math.exp(val[0]) if not isnan(val[0]) else 0.
            lx_norm.lx_minimum(x0=array([val[1], b_remapped]))
            if lx_norm.compare_lx():  # choose if Lx is lower than previous
                lx_norm.lx_first = lx_norm.lx_last
                logging.debug(
                    f"Last {parameters.COST_FUNCTION}-minimum: {lx_norm.lx_last}, "
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
                if f_synth < THRESHOLD:
                    f_n = append(f_n, f_synth)  # show < 16 kHz if applicable
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
