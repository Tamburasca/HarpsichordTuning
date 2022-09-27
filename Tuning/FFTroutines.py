from numpy import hanning, hamming, abs, ndarray
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, freqs, windows
from typing import Tuple
# internal
from Tuning.FFTaux import mytimer
from Tuning import parameters

hanning = hanning(M=parameters.SLICE_LENGTH)
hamming = hamming(M=parameters.SLICE_LENGTH)
apodization_gaussian = windows.gaussian(
    M=parameters.SLICE_LENGTH,
    std=parameters.SLICE_LENGTH / parameters.APODIZATION_GAUSS_SIGMA
)
t1 = rfftfreq(
    n=parameters.SLICE_LENGTH,
    d=1. / parameters.RATE
)
b, a = butter(
    N=parameters.F_ORDER,
    Wn=parameters.F_FILT,
    btype='highpass',
    analog=True)
_, h = freqs(
    b=b,
    a=a,
    worN=t1)


@mytimer
def fft(amp: ndarray) -> Tuple[ndarray, ndarray]:
    """
    performs FFT on a Gaussian apodized time series, where L = 7 sigma.
    High pass filter performed on frequency domain.
    :param amp: list float
        time series
    :return:
    list float
        frequency values of spectrum
    list float
        intensities
    """
    # apply apodization
    y_raw = rfft(apodization_gaussian * amp)
    # apply analog high pass Butterworth filter
    y_final = abs(h * y_raw)
    """
    if PSD then: 
    psd = 2. * noise power bandwidth * abs(y_raw)**2 / samples**2
    # noise power bandwidth = 2 or 1.81 for Hanning or Hamming windows, resp. 
    y_final = 2 * 2.0 * y_final ** 2 / len(amp)**2
    """

    return t1, y_final
