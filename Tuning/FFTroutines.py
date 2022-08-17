from numpy import hanning, hamming, abs, ndarray
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, freqs, windows
from typing import Tuple
# internal
from Tuning.FFTaux import mytimer
from Tuning import parameters

hanning = hanning(parameters.SLICE_LENGTH)
hamming = hamming(parameters.SLICE_LENGTH)
gaussian = windows.gaussian(
    M=parameters.SLICE_LENGTH,
    std=parameters.SLICE_LENGTH / parameters.APODIZATION_GAUSS_SIGMA
)
t1 = rfftfreq(parameters.SLICE_LENGTH, 1. / parameters.RATE)


@mytimer
def fft(amp: ndarray) -> Tuple[ndarray, ndarray]:
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
    y_raw = rfft(gaussian * amp)
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
    # noise power bandwidth = 2 or 1.81 for Hanning or Hamming windows, resp. 
    y_final = 2 * 2.0 * y_final ** 2 / len(amp)**2
    """

    return t1, y_final
