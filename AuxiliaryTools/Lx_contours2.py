"""
define first
export PYTHONPATH=${PYTHONPATH}:/home/ralf/pycharm-projects/Tuning:/home/ralf/pycharm-projects/Tuning/Tuning
"""

import argparse
import logging
import multiprocessing as mp
import signal
import sys
from re import findall, compile
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
from numpy import log10, arange, meshgrid, array, amin, amax, append

# internal
from Tuning.LxCostfunction3 import L1, L2
from Tuning.minimize_SLSQP_class import MinimizeSLSQP
from Tuning.parameters import INHARM, FREQUENCY_LOWER, FREQUENCY_UPPER

logging.getLogger().setLevel(logging.DEBUG)
plt.get_cmap('hsv')

# exact to b=1.e-4
FOUNDS = [415.0, 830.124478217012, 1245.4978506647046, 1661.2444090805334,
          2077.488259113232, 2494.353258899385, 2911.9629581560675,
          3330.4405379142177, 3749.908751014604, 4170.489863484847]
FOUNDS = [np.float64(625.0617492407505), np.float64(729.2145061075591), np.float64(1042.1720746787014),
          np.float64(1251.144900088567), np.float64(1984.5818972467348), np.float64(2089.5753454501623),
          np.float64(3041.213985921234), np.float64(3253.6318155002346)]


class Range(object):
    def __init__(self, scope: str):
        b, f = r"([\[\]])", r"([-+]?(?:\d*\.\d+|\d+\.?)(?:[Ee][+-]?\d+)?)"
        r = compile(f'^{b} ?{f} ?, ?{f} ?{b}$')
        try: i = self.i = list(findall(r, scope)[0])
        except IndexError: raise SyntaxError("Range error!")
        if float(i[1]) >= float(i[2]): raise ArithmeticError("Range error!")
        self.__lambda = "lambda item: {} {} item {} {}".format(
            *[i[1], {'[': '<=', ']': '<'}[i[0]], {']': '<=', '[': '<'}[i[3]], i[2]])
    def __eq__(self, item: float) -> bool: return eval(self.__lambda)(item)
    def __contains__(self, item: float) -> bool: return self.__eq__(item)
    def __iter__(self) -> Generator[object, None, None]: yield self
    def __str__(self) -> str: return '{}{}, {}{}'.format(*self.i)
    def __repr__(self) -> str: return self.__str__()


def init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main(
        minimizer,
        f0,
        b
) -> None:
    """

    :param minimizer:
    :param f0:
    :param b:
    :return:
    """
    initial = [f0, b]
    ind = FOUNDS

    if minimizer == 'L1':
        lx_norm = L1(ind)
    elif minimizer == 'L2':
        lx_norm = L2(ind)
    else:
        raise ValueError("Minimizer must be 'L1'|'L2'")

    mini = MinimizeSLSQP(minimizer)
    final = mini(ind=ind, f0=initial[0], b=initial[1])
    path = mini.path
    try:
        print(f"final value: {mini.res}")
    except AttributeError:
        pass

    # define grid
    # lower, upper limit f0
    # xm_l, xm_u = 0.995, 1.005
    xm_u = max(initial[0], final[0], amax(path, axis=0)[0]) * 1.01
    xm_l = min(initial[0], final[0], amin(path, axis=0)[0]) * 0.99
    x = arange(xm_l,
               xm_u,
               (xm_u - xm_l) / 2000.)
    # lower, upper limit B
    # ym_l, ym_u = -5.0, -3.7
    ym_u = max(log10(initial[1]), log10(final[1]), log10(amax(path, axis=0))[1]) + .5
    ym_l = min(log10(initial[1]), log10(final[1]), log10(amin(path, axis=0))[1]) - .5
    y = arange(ym_l,
               ym_u,
               (ym_u - ym_l) / 2000.)
    xgrid, ygrid = meshgrid(x, y)
    # xy = stack([xgrid, ygrid])

    res_lx = list()
    res_jac_f0 = list()
    res_jac_b = list()

    for y_data in y:
        tmp_lx = list()
        tmp_jac_f0 = list()
        tmp_jac_b = list()

        datadict = {}
        i = 0
        for x_data in x:
            datadict[i] = array([x_data, 10 ** y_data])
            i += 1
        inp = [v for _, v in sorted(datadict.items(), key=lambda item: item[0])]

        pool = mp.Pool(initializer=init_worker)
        try:
            result = pool.map(lx_norm.lx_minimum_der, inp)
        except KeyboardInterrupt:
            print("KeyboardInterrupt, terminating workers")
            pool.terminate()  # stop workers immediately
            pool.join()
            exit()
        else:
            pool.close()  # no more tasks
            pool.join()

        for tmp in result:
            tmp_lx.append(tmp[0])
            tmp_jac_f0.append(tmp[1][0])
            tmp_jac_b.append(tmp[1][1])
        res_lx.append(tmp_lx)
        res_jac_f0.append(tmp_jac_f0)
        res_jac_b.append(tmp_jac_b)

    res_lx = array(res_lx)
    res_jac_f0 = array(res_jac_f0)
    res_jac_b = array(res_jac_b)

    plt.figure(1)
    cs = plt.contour(xgrid, ygrid, res_lx, 35, cmap='hsv')
    plt.clabel(cs, inline=True, fontsize=10)
    plt.title(f"{minimizer} cost function")
    plt.xlabel("f0/Hz")
    plt.ylabel("B (log)")
    plt.scatter(path.T[0], log10(path.T[1]))
    plt.scatter(initial[0], log10(initial[1]), c='orange', label="initial")
    plt.scatter(final[0], log10(final[1]), c='black', label="final")
    plt.plot(
        append(append(array(initial[0]), path.T[0]), final[0]),
        append(append(array(log10(initial[1])), log10(path.T[1])), log10(final[1]))
    )
    plt.legend()

    plt.figure(2)
    cs = plt.contour(xgrid, ygrid, res_jac_f0, 30, cmap='hsv')
    plt.clabel(cs, inline=True, fontsize=10)
    plt.title(f"{minimizer} cost function - Jacobian derived to f0")
    plt.xlabel("f0/Hz")
    plt.ylabel("B (log)")
    plt.scatter(path.T[0], log10(path.T[1]))
    plt.scatter(initial[0], log10(initial[1]), c='orange', label="initial")
    plt.scatter(final[0], log10(final[1]), c='black', label="final")
    plt.plot(
        append(append(array(initial[0]), path.T[0]), final[0]),
        append(append(array(log10(initial[1])), log10(path.T[1])), log10(final[1]))
    )
    plt.legend()

    plt.figure(3)
    cs = plt.contour(xgrid, ygrid, res_jac_b, 20, cmap='hsv')
    plt.clabel(cs, inline=True, fontsize=10)
    plt.title(f"{minimizer} cost function - Jacobian derived to B")
    plt.xlabel("f0/Hz")
    plt.ylabel("B (log)")
    plt.scatter(path.T[0], log10(path.T[1]))
    plt.scatter(initial[0], log10(initial[1]), c='orange', label="initial")
    plt.scatter(final[0], log10(final[1]), c='black', label="final")
    plt.plot(
        append(append(array(initial[0]), path.T[0]), final[0]),
        append(append(array(log10(initial[1])), log10(path.T[1])), log10(final[1]))
    )
    plt.legend()

    try:
        plt.show()
    except KeyboardInterrupt:
        print("Keyboard interrupt")


if __name__ == "__main__":
    print(f"PYTHONPATH: {sys.path}")
    parser = argparse.ArgumentParser(
        description="Evaluate Lx-Minimizers")
    parser.add_argument(
        '-m',
        '--minimizer',
        type=str,
        help="Type of minimizer ['L1' | 'L2']",
        choices=["L1", "L2"]
    )
    parser.add_argument(
        '-f0',
        type=float,
        help="Base Frequency initial value [Hz]",
        choices=Range(f"[{FREQUENCY_LOWER}, {FREQUENCY_UPPER}]")
    )
    parser.add_argument(
        '-b',
        type=float,
        help="Inharmonicity initial value",
        choices=Range(f"[0., {INHARM}]")
    )

    main(
        minimizer=parser.parse_args().minimizer,
        f0=parser.parse_args().f0,
        b=parser.parse_args().b
    )
