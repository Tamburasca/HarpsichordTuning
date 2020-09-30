import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import signal
import timeit
import logging
import warnings
from .multiProcess_opt import ThreadedOpt
from .parameters import _debug, RATE, SIGMA, INHARM


format = "%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s"
logging.basicConfig(format=format,
                    level=logging.INFO,
                    datefmt="%H:%M:%S")
if _debug:
    logging.getLogger().setLevel(logging.DEBUG)


def baseline_als_optimized(y, lam, p, niter=10):
    """
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    """
    _start = timeit.default_timer()

    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
        # following early exit clause yields another 50 msec in speed
        if i > 0:
            if np.all(np.abs(z - z_last)) < 1.e-1:
                break
        z_last = z

    _stop = timeit.default_timer()
    logging.debug("baseline: " + str(i) + " iterations")
    logging.debug("time utilized for baseline [s]: " + str(_stop - _start))

    return z


def fft(amp, samples=None):
    """
    performs FFT on a Hanning apodized time series and a Gauss smoothing afterwards. High pass filter performed as
    well.

    :param amp: list float
        time series
    :param samples: int optional
         number of samples
    :return:
    list float
        frequency values of spectrum
    list float
        intensities
    """
    F_FILT = 55.  # high pass cutoff frequency @-3db
    F_ORDER = 2  # high pass Butterworth filter of order F_ORDER
    _start = timeit.default_timer()

    if not samples:
        samples = len(amp)
    hanning = np.hanning(samples)
    # gaussian = windows.gaussian(samples, std=0.1 * samples)
    y_raw = np.fft.rfft(hanning * amp)
    t1 = np.fft.rfftfreq(samples, 1. / RATE)
    # if PDS then y = 2. * RATE * np.abs(y_raw) ** 2 / samples
    # convolve with a Gaussian of width SIGMA
    spectrum = signal.convolve(in1=np.abs(y_raw),
                               in2=signal.gaussian(M=21, std=SIGMA),
                               mode='same')
    # high pass Butterworth filter
    b, a = signal.butter(F_ORDER, F_FILT, 'high', analog=True)
    _, h = signal.freqs(b, a, worN=t1)
    y_final = np.abs(h) * spectrum

    _stop = timeit.default_timer()
    logging.debug("time utilized for FFT [s]: " + str(_stop - _start))

    return t1, y_final


def peak(spectrum):
    """
    find peaks in frequency spectrum
    :param spectrum: list
        spectrum from FFT
    :return:
    list1
        list of peak heights
    list2
        list of frequencies
    """
    # max number of highest peaks
    NMAX = 25
    list1 = []
    list2 = []
    _start = timeit.default_timer()

    """
    DESCRIPTION This snippet computes the noise following the definition set forth by the Spectral Container 
                Working Group of ST-ECF, MAST and CADC.
                noise  = 1.482602 / sqrt(6) median(abs(2 flux_i - flux_i-2 - flux_i+2))
                values with padded zeros are skipped
    NOTES       The algorithm is an unbiased estimator describing the spectrum as a whole as long as
                * the noise is uncorrelated in wavelength bins spaced two pixels apart
                * the noise is Normal distributed
                * for large wavelength regions, the signal over the scale of 5 or more pixels can be approximated 
                by a straight line
                For most spectra, these conditions are met.
    REFERENCES  * ST-ECF Newsletter, Issue #42:
                www.spacetelescope.org/about/further_information/newsletters/html/newsletter_42.html
                * Software: www.stecf.org/software/ASTROsoft/DER_SNR/
    """
    # Values that are exactly zero (padded) are skipped
    flux = np.array(spectrum[np.where(spectrum != 0.0)])
    i = len(flux)
    std = 0.6052697 * np.median(abs(2.0 * flux[2:i - 2] - flux[0:i - 4] - flux[4:i]))
    logging.debug("noise estimate: " + str(std))

    # find peak according to prominence and remove peaks below threshold
    # prominences = signal.peak_prominences(x=spectrum, peaks=peaks)[0]
    # spectrum[peaks] == prominences with zero baseline
    peaks, _ = signal.find_peaks(x=spectrum,
                                 distance=12,  # min distance of peaks -> may need adjustment
                                 prominence=200 * std,  # sensitivity minus background
                                 width=(0, 16))  # max peak width -> may need adjustment
    nPeaks = len(peaks)
    # consider NMAX highest, sort key = amplitude descending
    if nPeaks != 0:
        list1, list2 = (list(t) for t in zip(*sorted(zip(spectrum[peaks], peaks), reverse=True)))
        del list2[NMAX:]
        del list1[NMAX:]
    # re-sort with sort key = frequency ascending
        list2, list1 = (list(t) for t in zip(*sorted(zip(list2, list1))))

    _stop = timeit.default_timer()
    logging.debug("Peaks found: " + str(nPeaks))
    logging.debug("Time for peak finding [s]: " + str(_stop - _start))

    return list2, list1


def harmonics(amp, freq, ind, height=None):
    """
    :param amp: ndarray
        amplitudes of FFT transformed spectrum
    :param freq: ndarray
        freqencies of FFT transformed spectrum (same dimension of amp
    :param ind: ndarray
        peak positions found in FFT spectrum (NMAX highest)
    :param height: ndarray
        peak height found in FFT spectrum (NMAX highest)
    :return:
    ndarray
        positions of first 8 partials as found in fit
    """
    warnings.filterwarnings('error', message='', category=Warning)

    NPARTIAL = 11
    initial = []
    logging.debug("ind: " + str(ind))
    logging.debug("height: " + str(height))

    _start = timeit.default_timer()

    for i_ind, j_ind in zip(ind, ind[1:]):
        # Loop through the partials up to NPARTIAL
        for m, k in zip(range(1, NPARTIAL), range(2, NPARTIAL)):
            tmp = (j_ind * m / k) ** 2
            try:
                b = (tmp - i_ind ** 2) / ((k * i_ind) ** 2 - tmp * m ** 2)
            except Warning:
                logging.info("devideByZero: discarded value in harmonics finding")
                # skip this one
                continue
            # INHARM is also used in boundaries in multiProcess.py
            if 0 <= b < INHARM:
                f_fundamental = i_ind / (m * np.sqrt(1. + b * m ** 2))
                logging.debug("partial: {0:d} {1:d} lower: {2:7.2f} upper: {3:7.2f} "
                              "b: {4:0.6f} fundamental: {5:7.2f}".
                              format(m, k, i_ind, j_ind, b, f_fundamental))
                # pump it all to the minimizer and let him decide, what's best
                if f_fundamental < 27.5:
                    continue
                initial.append(tuple((f_fundamental, b)))

    warnings.resetwarnings()

    if not initial and len(ind) > 0:  # if nothing was found, give it a shot with the strongest peak
        _, list2 = (list(t) for t in zip(*sorted(zip(height, ind), reverse=True)))  # re-sort with key=height desc
        initial.append(tuple((list2[0], 0.)))
    opt = ThreadedOpt(amp, freq, initial, height)
    opt.run
    logging.info("Best result [f0, B]=" + str(opt.best_x))

    # prepare for displaying vertical bars, and key finding etc.
    f_n = np.array([])
    if opt.best_x is not None:
        for n in range(1, 11):
            f_n = np.append(f_n, opt.best_x[0] * n * np.sqrt(1. + opt.best_x[1] * n**2))

    _stop = timeit.default_timer()
    logging.debug("time utilized for minimizer [s]: " + str(_stop - _start))

    return f_n
