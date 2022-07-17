from numpy import abs, average, median, append, array, insert
from scipy.signal import find_peaks
import logging

from Tuning.multiProcess_opt_gauss import ThreadedOpt
from Tuning.FFTaux import mytimer
from Tuning import parameters as P


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
    def __init__(self, flux: array):
        i = len(flux)
        self.__flux = abs(2. * flux[2:i - 2] - flux[0:i - 4] - flux[4:i])
        # padding two value to prepend and two to append
        self.__b = insert(self.__flux, 0, [flux[0]]*2)
        self.__b = append(self.__b, [self.__flux[-1]]*2)
        self.__l = len(self.__b)

    def __call__(self, value: int, width: int = 50):
        low = value - width if value - width >= 0 else 0
        high = value + width if value + width < self.__l else self.__l - 1
        return average(self.__b[low:high])

    def total(self):
        return 0.6052697 * median(self.__flux)


@mytimer("peak finding (subtract time consumed for Parabola fits)")
def peak(frequency, spectrum):
    """
    find peaks in frequency spectrum
    :param frequency: list
        frequencies from FFT
    :param spectrum: list
        spectrum amplitudes from FFT
    :return:
        list (float) of tuples with peak frequencies and corresponding heights
        (no baseline subtracted)
    """
    listf = list()

    noise = Noise(flux=spectrum)
    std = noise.total()
    logging.debug("Noise estimate: {0}".format(std))

    # find peak according to prominence and remove peaks below threshold
    # spectrum[peaks] = "prominences with baseline zero"
    peaks, properties = find_peaks(x=spectrum,
                                   # min distance between two peaks
                                   distance=P.DISTANCE,
                                   # sensitivity minus background
                                   # prominence=P.NOISE_LEVEL * std,
                                   # peak width
                                   width=P.WIDTH)
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
    listtup = [item for item in listtup if
               item[1] > P.NOISE_LEVEL * std]
    listtup.sort(key=lambda x: x[1], reverse=True)
    del listtup[P.NMAX:]

    if len(listtup) != 0:
        # run Gaussfits to the lines found subsequently
        opt = ThreadedOpt(freq=frequency,
                          amp=spectrum,
                          initial=listtup)
        listf = opt()
        sortedf = sorted(listf, key=lambda x: x[0])
        for line in sortedf:
            logging.debug(
                "Position: {0:10.4f} Hz, "
                "Height (arb. units): {1:.2e}, "
                "FWHM: {2:5.2f} Hz".format(line[0], line[1], 2.354 * line[2]))
    logging.debug("Peaks considered: {}".format(len(listf)))

    return listf
