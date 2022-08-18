from numpy import log, sqrt, exp
from operator import itemgetter
import logging
from typing import List, Tuple, Union
# internal
from Tuning.FFTaux import mytimer
from Tuning import parameters


class ThreadedOpt(object):

    def __init__(self, freq: List, amp: List, initial: List[Tuple]):
        """
        :param freq: list of floats
            frequencies of the FFT
        :param amp:  list of floats
            amplitudes of the FFT
        :param initial: list of float tuple
            frequencies and corresponding heights of peaks found
        """
        self.__amp = amp
        self.__freq = freq
        self.__num_threads = len(initial)
        self.__x = list(map(itemgetter(0), initial))
        self.__y = list(map(itemgetter(1), initial))

    @mytimer("Parabola interpolation")
    def __call__(self) -> List[List]:
        peaks = list()
        logging.debug("Number of peaks: {0}".format(self.__num_threads))
        for thread_id in range(self.__num_threads):
            result = self.__fitting(thread_id)
            if result:
                peaks.append(result)

        return peaks

    def __fitting(self, thread_id: int) -> Union[List, None]:
        """
        Each thread goes through this.
        :param thread_id: int
            thread id = 0, 1, 2, ..., len(initial)-1
        :return: list[ctr, height, fwhm] | None
        """
        x = self.__x[thread_id]  # frequency bin with peak
        # do not exceed the array on either side
        if (x - 1) < 0 or (x + 2) > len(self.__freq):
            logging.warning("Fit center value out of window: peak disregarded!")
            return None

        f = self.__freq[x - 1:x + 2]  # frequencies: one neighbor on either side
        test = self.__amp[x - 1:x + 2]  # amplitudes
        if test[0] > test[1] or \
                test[2] > test[1] or \
                (test[1] * test[1]) < (test[0] * test[2]):
            logging.warning("Convolution requirement violated: peak disregarded!")
            return None

        a = log(self.__amp[x - 1:x + 2])  # log amplitudes in Fourier space
        """
        EUROPEAN ORGANIZATION FOR NUCLEAR RESEARCH / ORGANISATION EUROPEENNE POUR LA RECHERCHE NUCLEAIRE
        CERN – AB DIVISION
        AB-Note-2004-021 BDI, February 2004
        by M. Gasior, J.L. Gonzalez
        three-node interpolation of the logarithm of a Gaussion to a parabola
        https://mgasior.web.cern.ch/pap/FFT_resol_note.pdf
        for reference: three-node interpolation
        https://stackoverflow.com/questions/4039039/fastest-way-to-fit-a-parabola-to-set-of-points
        """
        offset = .5 * (a[2] - a[0]) / (
                2. * a[1] - a[0] - a[2]) * parameters.RATE / parameters.SLICE_LENGTH
        ctr = f[1] + offset
        dilation = (a[0] - a[1]) / ((f[0] - ctr) ** 2 - (f[1] - ctr) ** 2)
        fwhm = 2. * sqrt(-.6931 / dilation)
        height = exp(a[0] - dilation * (f[0] - ctr) ** 2)

        return list([ctr, height, fwhm])
