import logging
import math
from operator import itemgetter

from numpy import sqrt, mean, append, array
from numpy.typing import NDArray

# internal
import parameters
from FFTaux import mytimer
from minimize_SLSQP_class import MinimizeSLSQP

I_MAX = int(16_000 / parameters.FREQUENCY_LOWER)


def l2min_new(
        ind: list,
        f0: float,
        b: float,
) -> float:
    """
    Note: L2 for testing and reference, not currently used in minimization.
    returns the cost function for a regression on the L2 norm
    l2 = sum( ( ( f_i(measured) - f_i(calculated) ) / f_i(measured) )**2 )
    where f_i(measured) is considered to its nearest neighbor either
    f_i(calculated) or f_i+1(calculated)
    :param ind: list - measured resonance frequencies as from peaks (FFT)
    after being cleansed, duplicates removed, etc.
    :param f0: float - base frequency
    :param b: float - inharmonicity, such that f = i * f0 * sqrt(1. + b * i**2)
    :return: float - l2 cost function
    """
    l2 = 0.  # l2 cost function
    j = 1
    # loop over peaks found
    for found in ind:
        fl = j * f0 * sqrt(1. + b * j ** 2)
        for i in range(j, I_MAX):
            fh = (i + 1) * f0 * sqrt(1. + b * (i + 1) ** 2)
            j = i
            if found < fl and i == 1:
                diff = fl - found
                l2 += diff * diff / found / found
                break
            elif fl <= found < fh:
                if (found - fl) < (fh - found):
                    diff = found - fl
                else:
                    diff = fh - found
                l2 += diff * diff / found / found
                break
            else:
                fl = fh

    return l2


def l1min_new(
        ind: list,
        f0: float,
        b: float,
) -> float:
    """
    returns the cost function for a regression on the L1 norm
    l1 = sum( abs( f_i(measured) - f_i(calculated) ) / f_i(measured) )
    where f_i(measured) is considered to its nearest neighbor either
    f_i(calculated) or f_i+1(calculated)
    :param ind: list - measured resonance frequencies as from peaks (FFT)
    after being cleansed, duplicates removed, etc.
    :param f0: float - base frequency
    :param b: float - inharmonicity, such that f = i * f0 * sqrt(1. + b * i**2)
    :return: float - l1 cost function
    """
    l1 = 0.  # l1 cost function
    j = 1
    # loop over peaks found
    for found in ind:
        fl = j * f0 * sqrt(1. + b * j ** 2)
        for i in range(j, I_MAX):
            fh = (i + 1) * f0 * sqrt(1. + b * (i + 1) ** 2)
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
    l1: dict[tuple[int, float], list[float]] = dict()
    f_n = list()

    # sort by frequency asc. and make list of indices (positions) and heights
    peaks.sort(key=lambda x: x[0])
    ind = list(map(itemgetter(0), peaks))
    height = list(map(itemgetter(1), peaks))
    logging.debug("ind: " + str(ind))
    logging.debug("height: " + str(height))

    nex = 1
    # loop through all peaks found (ascending, nested loops)
    for i in range(0, len(ind)):
        for j in range(i + 1, len(ind)):
            # loop through all combinations of partials up to NPARTIAL
            for m in range(nex, parameters.NPARTIAL):
                for k in range(m + 1, parameters.NPARTIAL):
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
                            if (element[3] == initial[-1][3]
                                    and element[0] == initial[-1][0]):
                                # remove previous doublette on upper frequency and lower partial
                                # print("removed doublette", initial[-1], element)
                                initial.pop()
                        initial.append(element)
        nex += 1  # increase lower partial for next higher peak found

    if initial:
        l1_min = float('inf')

        for item in initial:  # if found any partial combinations
            t = (item[0], item[2])
            initial_log = [
                math.log(item[4]) if item[4] > 0.0 else float('nan'),
                item[5]
            ]
            if t not in l1:
                l1[t] = list()
            l1[t].append(initial_log)  # type: ignore
            print(l1)
            logging.debug(
                "partials: {0:2d} {1:2d} lower: {2:10.4f} upper: {3:10.4f} "
                "B: {4: .1e} fundamental: {5:10.4f}".format(*item))

        for key, val in l1.items():
            arrays = [array(x) for x in val]
            initial_av = [mean(k) for k in zip(*arrays)]
            t_new = l1min_new(
                ind=ind,
                f0=float(initial_av[1]),
                b=math.exp(initial_av[0]) if not math.isnan(initial_av[0]) else 0.)
            if t_new < l1_min:
                l1_min = t_new
                logging.debug("Last L1 minimum: {}".format(l1_min))
                selected = array(
                    list(
                        filter(lambda x: (x[0], x[2]) == key, initial))
                )
                base_frequency = float(initial_av[1])
                inharmonicity = math.exp(initial_av[0]) if not math.isnan(initial_av[0]) else 0.

        if (parameters.FREQUENCY_LOWER
                < base_frequency
                < parameters.FREQUENCY_UPPER):
            identified = select_list(selected=selected)
            base_frequency_final, inharmonicity_final = (
                MinimizeSLSQP(norm=parameters.COST_FUNCTION)(
                    ind=identified,
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
