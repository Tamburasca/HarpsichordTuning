from numpy import hanning, hamming, sqrt, abs, average, median, append, array,\
    mean, insert
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, freqs, find_peaks, peak_prominences
from math import gcd
import logging
from operator import itemgetter

from .multiProcess_opt import ThreadedOpt
from .FFTaux import mytimer
from Tuning import parameters

logging.basicConfig(format=parameters.myformat,
                    level=logging.INFO,
                    datefmt="%H:%M:%S")
if parameters.DEBUG:
    logging.getLogger().setLevel(logging.DEBUG)

hanning = hanning(parameters.SLICE_LENGTH)
hamming = hamming(parameters.SLICE_LENGTH)
t1 = rfftfreq(parameters.SLICE_LENGTH, 1./parameters.RATE)


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


@mytimer
def fft(amp):
    """
    performs FFT on a Hanning apodized time series. High pass filter performed
    on frequency domain.
    :param amp: list float
        time series
    :return:
    list float
        frequency values of spectrum
    list float
        intensities
    """
    y_raw = rfft(hamming * amp)
    """
    convolve with a Gaussian of width SIGMA
    y_raw = convolve(in1=abs(y_raw),
                     in2=gaussian(M=11, std=1),
                     mode='same')
    """
    # analog high pass Butterworth filter
    # ToDo: would a digital filter be faster?
    b, a = butter(N=parameters.F_ORDER,
                  Wn=parameters.F_FILT,
                  btype='high',
                  analog=True)
    _, h = freqs(b=b,
                 a=a,
                 worN=t1)
    y_final = abs(h * y_raw)
    """
    if PSD then: 
    psd = 2. * noise power bandwidth * abs(y_raw)**2 / samples**2
    # noise power bandwidth = 1.5 for Hanning window
    y_final = 2 * 1.5 * y_final ** 2 / len(amp)**2
    """

    return t1, y_final


@mytimer("peak finding (subtract time consumed for Gauss fits)")
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
    # prominences = signal.peak_prominences(x=spectrum, peaks=peaks)[0]
    # spectrum[peaks] == prominences with zero baseline
    peaks, properties = find_peaks(x=spectrum,
                                   # min distance between two peaks
                                   distance=parameters.DISTANCE,
                                   # sensitivity minus background
                                   prominence=parameters.NOISE_LEVEL * std,
                                   # peak width
                                   width=parameters.WIDTH)
    # peak_height = peak_prominences(spectrum, peaks)[0]
    # print(spectrum[peaks], peak_height)
    # print(peaks, properties)
    npeaks = len(peaks)
    logging.debug("Peaks found: {0}".format(npeaks))
#    peaks = [i for i in peaks if spectrum[i] > 5. * noise(i, 50)]

    # consider NMAX highest, sort key = amplitude descending
    if npeaks != 0:
        listtup = list(zip(peaks, spectrum[peaks]))
        listtup.sort(key=lambda x: x[1], reverse=True)
        del listtup[parameters.NMAX:]

        # run Gaussfits to the lines found in different processes
        opt = ThreadedOpt(freq=frequency,
                          amp=spectrum,
                          initial=listtup)
        listf = opt.run()
        sortedf = sorted(listf, key=lambda x: x[0])
        for line in sortedf:
            logging.debug(
                "Position: {0:e} Hz, "
                "Height (arb. Units): {1:e}, "
                "FWHM: {2:e} Hz".format(line[0], line[1], 2.354 * line[2]))
    logging.debug("Peaks considered: " + str(len(listf)))

    return listf


def bisection(vector, value):
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


def l1_fit(x0, fo):
    """
    returns the cost function for a Lasso regression
    :param x0: array - [f0, b] such that f = i * x0[0] * sqrt(1. + x0[1] * i**2)
    :param fo: array - measured resonance frequencies as from FFT
    :return: float - l1 cost function
    """
    if not fo or not x0:
        return None
    freq = list()
    fmax = max(fo)
    for i in range(1, 640):
        f = i * x0[0] * sqrt(1. + x0[1] * i * i)
        freq.append(f)
        if f > fmax:
            break  # exit if superseded max. frequency measured to save time
    num_freq = len(freq)
    l1 = 0.  # l1 cost function
    for found in fo:
        idx = bisection(freq, found)
        # L1 norm
        if idx == -1:
            l1 += abs(found - freq[0])  # <min frequency
        elif idx == num_freq:
            l1 += abs(found - freq[num_freq-1])  # >max frequency
        else:
            # consider the closest candidate of neighbors
            l1 += min(abs(found - freq[idx]), abs(found - freq[idx + 1]))

    return l1


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

    # sort by frequency ascending
    peaks.sort(key=lambda x: x[0])
    ind = list(map(itemgetter(0), peaks))
    height = list(map(itemgetter(1), peaks))
    logging.debug("ind: " + str(ind))
    logging.debug("height: " + str(height))

    # loop through the combination of partials up to NPARTIAL
    for m in range(1, parameters.NPARTIAL):
        for k in range(m+1, parameters.NPARTIAL):
            # loop through all peaks found (ascending, neested loops)
            for i in range(0, len(ind)):
                for j in range(i+1, len(ind)):
                    tmp = ((ind[j] * m) / (ind[i] * k))**2
                    try:
                        b = (tmp - 1.) / (k**2 - tmp * m**2)
                    except ZeroDivisionError:
                        logging.info("devideByZero: "
                                     "discarded value in harmonics finding")
                        continue
                    if -0.0001 < b < parameters.INHARM:
                        # allow also negative b value > -0.0001 for uncertainty
                        # in the line fitting
                        f_fundamental = ind[i] / (m * sqrt(1. + b * m ** 2))
                        logging.debug(
                            "partial: {0:2d} {1:2d} "
                            "lower: {2:9.4f} upper: {3:9.4f} "
                            "b: {4: .1e} fundamental: {5:9.4f}"
                            .format(m, k, ind[i], ind[j], b, f_fundamental))
                        # always b >= 0
                        initial.append(
                            [m, k, ind[i], ind[j], max(b, 0.), f_fundamental]
                        )
                        if gcd(m, k) == 1:
                            if m not in l1:
                                # create keys in dict with empty list values
                                l1[m] = list()
                        break
                break

    """
    ToDo: this section needs some modification 
    disregard fundamentals for records, where both partials have a common 
    divisor (gcd). Consider all fundamentals with no common divisor only.
    """
    f_n = list()
    if initial:
        f0, b = list(), list()
        if len(l1) > 1:
            # if more than one lower partial with gcd=1
            for key in l1:
                for dat in filter(lambda x: x[0] == key, initial):
                    # Add l1 values for same lower partial
                    l1[key].append(l1_fit([dat[5], dat[4]], ind))
            # when at least one combination with gcd = 1
            for key in l1:
                # l1 being averaged for each lower partial
                l1[key] = mean(l1[key])
            # find lower partial with min l1
            for dat in list(
                    filter(lambda x: x[0] == min(l1, key=l1.get), initial)):
                f0.append(dat[5])
                b.append(dat[4])
        elif len(l1) == 1:
            # if only one lower partial with gcd=1
            for dat in list(
                    filter(lambda x: x[0] == list(l1.keys())[0], initial)):
                f0.append(dat[5])
                b.append(dat[4])
        if not f0:
            # if no gcd=1 found, take first entry for the lowest partial
            for dat in list(filter(lambda x: x[0] == initial[0][0], initial)):
                f0.append(dat[5])
                b.append(dat[4])
        # disregard f0<26.5 Hz = A0
        if average(f0) > 26.5:
            for n in range(1, parameters.NPARTIAL):
                f_n = append(f_n, average(f0) * n * sqrt(
                    1. + average(b) * n ** 2))
            logging.info(
                "Best result: f_1 = {0:.2f} Hz, B = {1:.1e}".format(
                        f_n[0], average(b)))
    elif not initial and len(ind) > 0:
        # if fundamental could not be calculated through at least two lines,
        # give it a shot with the strongest peak being found
        peaks.sort(key=lambda x: x[1], reverse=True)  # sort by amplitude desc
        f1 = list(map(itemgetter(0), peaks))[0]
        # disregard f0 < 26.5 Hz = A0
        if f1 > 26.5:
            f_n.append(f1)
            logging.info("Best result: f_1 = {0:.2f} Hz, B = {1:.1e}".format(
                f1, 0.))

    return f_n
