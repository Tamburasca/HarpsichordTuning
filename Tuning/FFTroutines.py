import numpy as np
from scipy import signal
import timeit
import logging
import math
from operator import itemgetter
from .multiProcess_opt import ThreadedOpt
from Tuning import parameters

logging.basicConfig(format=parameters.myformat,
                    level=logging.INFO,
                    datefmt="%H:%M:%S")
if parameters.DEBUG:
    logging.getLogger().setLevel(logging.DEBUG)

hanning = np.hanning(parameters.SLICE_LENGTH)
t1 = np.fft.rfftfreq(parameters.SLICE_LENGTH,
                     1. / parameters.RATE)


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
    _start = timeit.default_timer()

    y_raw = np.fft.rfft(hanning * amp)
    """convolve with a Gaussian of width SIGMA
    y_raw = signal.convolve(in1=np.abs(y_raw),
                               in2=signal.gaussian(M=11, std=1),
                               mode='same')
    """
    # analog high pass Butterworth filter
    # ToDo: would a digital filter be faster?
    b, a = signal.butter(N=parameters.F_ORDER,
                         Wn=parameters.F_FILT,
                         btype='high',
                         analog=True)
    _, h = signal.freqs(b=b,
                        a=a,
                        worN=t1)
    y_final = np.abs(h * y_raw)
    """if PSD then 
    psd = 2. * np.abs(y_raw)**2 / samples**2 * noise power bandwidth
    # noise power bandwidth = 1.5 for Hanning window
    y_final = 2 * 1.5 * y_final ** 2 / len(amp)**2
    """
    _stop = timeit.default_timer()
    logging.debug("time utilized for FFT [s]: " + str(_stop - _start))

    return t1, y_final


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
    _start = timeit.default_timer()
    listf = list()

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
    """
    # Values that are exactly zero (padded) are skipped
    flux = np.array(spectrum[np.where(spectrum != 0.0)])
    i = len(flux)
    std = 0.6052697 * np.median(
        abs(2.0 * flux[2:i - 2] - flux[0:i - 4] - flux[4:i]))
    logging.debug("noise estimate: " + str(std))

    # find peak according to prominence and remove peaks below threshold
    # prominences = signal.peak_prominences(x=spectrum, peaks=peaks)[0]
    # spectrum[peaks] == prominences with zero baseline
    peaks, _ = signal.find_peaks(x=spectrum,
                                 # min distance between two peaks
                                 distance=parameters.DISTANCE,
                                 # sensitivity minus background
                                 prominence=parameters.NOISE_LEVEL * std,
                                 # peak width
                                 width=parameters.WIDTH)
    npeaks = len(peaks)
    logging.debug("Peaks found: " + str(npeaks))
    _stop = timeit.default_timer()
    logging.debug("Time for peak finding [s]: " + str(_stop - _start))

    _start = timeit.default_timer()
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

    logging.debug("Peaks considered: " + str(len(listf)))
    _stop = timeit.default_timer()
    logging.debug("Time for peak fitting [s]: " + str(_stop - _start))

    return listf


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
    _start = timeit.default_timer()
    initial = []

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
                    tmp = (ind[j] * m / k) ** 2
                    try:
                        b = (tmp - ind[i] ** 2) / \
                            ((k * ind[i]) ** 2 - tmp * m ** 2)
                    except ZeroDivisionError:
                        logging.info("devideByZero: "
                                     "discarded value in harmonics finding")
                        continue
                    if -0.0001 < b < parameters.INHARM:
                        # allow also negative b value > -0.0001 for uncertainty
                        # in line fitting
                        if b < 0:
                            b = 0
                        f_fundamental = ind[i] / (m * np.sqrt(1. + b * m ** 2))
                        logging.debug(
                            "partial: {0:2d} {1:2d} "
                            "lower: {2:9.4f} upper: {3:9.4f} "
                            "b: {4:.1e} fundamental: {5:9.4f}"
                            .format(m, k, ind[i], ind[j], b, f_fundamental))
                        initial.append([m, k, ind[i], ind[j], b, f_fundamental])
                        break
                break

    """
    disregard fundamentals for those records, where both partials have a 
    greatest common divisor (gcd). Consider the first fundamental found, 
    where gcd == 1
    """
    f_n = list()
    if initial:
        f0, b = list(), list()
        for item in reversed(initial):
            i = item[0]
            if math.gcd(item[0], item[1]) == 1:
                tmp = list(filter(lambda x: x[0] == item[0], initial))
                for dat in tmp:
                    f0.append(dat[5])
                    b.append(dat[4])
                break
        if not f0:
            # when no result found take last entry for lowest partial
            tmp = list(filter(lambda x: x[0] == i, initial))
            for dat in tmp:
                f0.append(dat[5])
                b.append(dat[4])
        logging.info("Best result: f_0 = {0} Hz, B = {1}".format(
            str(np.average(f0)),
            str(np.average(b)))
        )
        for n in range(1, parameters.NPARTIAL):
            f_n = np.append(f_n,
                            np.average(f0) * n * np.sqrt(
                                1. + np.average(b) * n ** 2))
    elif not initial and len(ind) > 0:
        # if fundamental could not be calculated through at least two lines,
        # give it a shot with the strongest peak found
        peaks.sort(key=lambda x: x[1], reverse=True)  # sort by amplitude desc
        f1 = list(map(itemgetter(0), peaks))[0]
        if f1 >= 27.0:
            f_n.append(f1)

    _stop = timeit.default_timer()
    logging.debug("time utilized for harmonics [s]: " + str(_stop - _start))

    return f_n
