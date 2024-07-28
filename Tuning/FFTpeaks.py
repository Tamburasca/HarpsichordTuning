from __future__ import annotations
from numpy import abs, average, median, append, insert, log, sqrt, exp
from scipy.signal import find_peaks
from operator import itemgetter
import logging
from typing import List, Tuple
from numpy.typing import NDArray
# internal
from FFTaux import mytimer
import parameters


class Noise:
    """
    DESCRIPTION:
    This snippet computes the noise following the definition set forth by the
    Spectral Container. Working Group of ST-ECF, MAST and CADC.
    noise  = 1.482602 / sqrt(6) median(abs(2 flux_i - flux_i-2 - flux_i+2))
    values with padded zeros are skipped
    NOTES
    The algorithm is an unbiased estimator describing the spectrum as a whole as
    long as
    * the noise is uncorrelated in wavelength bins spaced two pixels apart
    * the noise is Normal distributed
    * for large wavelength regions, the signal over the scale of 5 or more
    pixels can be approximated by a straight line
    For most spectra, these conditions are met.
    REFERENCES  * Software: www.stecf.org/software/ASTROsoft/DER_SNR/

    Comment: __call__ not utilized, as windows can be used
    """
    def __init__(self, flux: NDArray):
        i = len(flux)
        self.__flux = abs(2. * flux[2:i - 2] - flux[0:i - 4] - flux[4:i])
        # padding two value to prepend and two to append
        self.__b = insert(self.__flux, 0, [flux[0]]*2)
        self.__b = append(self.__b, [self.__flux[-1]]*2)
        self.__l = len(self.__b)

    def __call__(
            self,
            value: int,
            width: int = 50
    ) -> float:
        low = value - width if value - width >= 0 else 0
        high = value + width if value + width < self.__l else self.__l - 1
        return average(self.__b[low:high])

    def total(self) -> float:
        return 0.6052697 * median(self.__flux)


@mytimer("Gaussian Convolution")
def gaussian_convolution(
        freq: List,
        amp: List,
        initial: List[Tuple]
) -> List[List[float]]:
    """
    :param freq: list of floats
        frequencies of the FFT
    :param amp:  list of floats
        amplitudes of the FFT
    :param initial: list of float tuple
        frequencies and corresponding heights of peaks found
    """
    _amp = amp.copy()
    _freq = freq.copy()
    _x = list(map(itemgetter(0), initial))
    # _y = list(map(itemgetter(1), initial))  # obsolete
    num_threads = len(initial)
    peaks: List = []
    logging.debug("Number of peaks: {0}".format(num_threads))

    def _fitting(ids: int) -> List | None:
        """
        Each thread goes through this.
        :param ids: int, thread id = 0, 1, 2, ..., len(initial)-1
        :return: list[ctr, height, fwhm] | None
        """
        x = _x[ids]  # frequency bin with peak
        # do not exceed the array on either side
        if (x - 1) < 0 or (x + 2) > len(_freq):
            logging.warning("Fit center value out of window: peak disregarded!")
            return None

        f = _freq[x - 1:x + 2]  # frequencies: one neighbor on either side
        test = _amp[x - 1:x + 2]  # amplitudes
        if (test[0] > test[1] or test[2] > test[1]
                or (test[1] * test[1]) < (test[0] * test[2])):
            logging.warning(
                "Convolution requirement violated: peak disregarded!"
            )
            return None
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
        a = log(_amp[x - 1:x + 2])  # log amplitudes in Fourier space
        offset = (
                .5 * parameters.RATE / parameters.SLICE_LENGTH *
                (a[2] - a[0]) / (2. * a[1] - a[0] - a[2])
        )
        ctr = f[1] + offset
        dilation = (a[0] - a[1]) / ((f[0] - ctr) ** 2 - (f[1] - ctr) ** 2)
        fwhm = 2. * sqrt(-.6931 / dilation)
        height = exp(a[0] - dilation * (f[0] - ctr) ** 2)

        return list([ctr, height, fwhm])
        # end embedded function

    for i in range(num_threads):
        result = _fitting(ids=i)
        if result:
            peaks.append(result)

    return peaks


@mytimer("peak finding (subtract time consumed for Gaussian Convolution)")
def peak(
        frequency: NDArray,
        spectrum: NDArray,
        baseline: [None, NDArray],
        std: [None, NDArray],
        noise_level: float
) -> List[List[float]]:
    """
    find peaks in frequency spectrum
    :param frequency: NDArray
        frequencies from FFT
    :param spectrum: NDArray
        spectrum amplitudes from FFT
    :param baseline [None, NDArray]
        baseline of the audio signal in frequency domain after being averaged
        in silence, consider if baseline is not None
    :param std [None, NDArray]
        standard deviation of the baseline of the audio signal in frequency
        domain after being averaged in silence
    :param noise_level: float
        noise level as threshold for peak detection
    :return:
        list (float) of tuples with peak frequencies and corresponding heights
        (no baseline subtracted)
    """
    listf: List = []

    noise = Noise(flux=spectrum)
    noise_total = noise.total()
    logging.debug("Noise estimate: {0}".format(noise_total))

    # find peak according to prominence and remove peaks below threshold
    # spectrum[peaks] = "prominences with baseline zero"
    peaks, properties = find_peaks(
        x=spectrum,
        # min distance between two peaks
        distance=parameters.DISTANCE,
        # sensitivity minus background
        # prominence=parameters.NOISE_LEVEL * noise_total,
        # peak width
        width=parameters.WIDTH
    )
    # print(peaks, properties['left_ips'], properties['right_ips'])
    # avaraged background of both sides
    left = [int(i) for i in properties['left_ips']]
    right = [int(i + 1) for i in properties['right_ips']]
    # subtract background
    corrected = spectrum[peaks] - (spectrum[left] + spectrum[right]) / 2
    logging.debug("Peaks found: {0}".format(len(peaks)))
    listtup = list(zip(peaks, corrected))
    # sort out peaks below threshold and consider NMAX highest,
    # sort key = amplitude descending
    if baseline is None:
        listtup = \
            [item for item in listtup if item[1] > noise_level * noise_total]
    else:
        # ToDo: needs to be tested
        listtup = [item for item in listtup if item[1] > 20. * std[item[0]]]
    listtup.sort(key=lambda x: x[1], reverse=True)
    del listtup[parameters.NMAX:]

    if len(listtup) != 0:
        # run Gaussfits to the lines found
        listf = gaussian_convolution(freq=frequency,
                                     amp=spectrum,
                                     initial=listtup)
        sortedf = sorted(listf, key=lambda x: x[0])
        for line in sortedf:
            logging.debug(
                "Position: {0:10.4f} Hz, "
                "Height (arb. units): {1:.2e}, "
                "FWHM: {2:5.2f} Hz".format(line[0], line[1], line[2]))
    logging.debug("Peaks considered: {}".format(len(listf)))

    return listf
