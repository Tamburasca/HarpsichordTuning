"""logging format"""
myformat = "%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s"

"""audio sampling rate in kHz derived from driver -> hardware & sound
do not modify!"""
RATE: int = 44100

"""Factor to multiply with the slice length: [1|2]"""
FACTOR: int = 1

"""size of FFT slice: 32768 samples/slice: => 0.743 sec sampling time/slice.
Corrected for the sample size."""
SLICE_LENGTH: int = 32768 * FACTOR

"""no of samples (initial value) by which each slice is shifted with regard 
to its previous, can be adjusted by +/-1024 samples through hotkeys crtl-j/k.
Corrected for the sample size."""
SLICE_SHIFT: int = 16384

"""F_FILT: high pass cutoff frequency [Hz]"""
F_FILT: float = 100.

"""F_ORDER: order of high pass Butterworth filter"""
F_ORDER: int = 2

"""noise level needs to be adjusted such, that there are no peaks detected with 
no key pressed (silence). The threshold of peaks being regarded is calculated 
as NOISE_LEVEL * noise (calculated over the entire spectrum)"""
NOISE_LEVEL: float = 50.

"""factor multiplied with the standard deviation and added to the mean noise 
background, after measurement of the noise background (may be > 6.)"""
FACTOR_STANDARD_DEV_NOISE = 7.

"""Required minimal horizontal distance (>= 1) in samples between neighbouring 
peaks. Smaller peaks are removed first until the condition is fulfilled for 
all remaining peaks. Corrected for the sample size."""
DISTANCE: int = 16 * FACTOR

"""(min, max) width of peak in channels to identify a peak as such. Corrected 
for the sample size."""
WIDTH: tuple = (1 * FACTOR, 8 * FACTOR)

"""Sigma of the Gauss apodization window, where the length of slice
L = APODIZATION_GAUSS_SIGMA * sigma
do not modify"""
APODIZATION_GAUSS_SIGMA: float = 7.

"""max. inharmonicity of strings considered (harpsichord, piano, ...)"""
INHARM: float = 0.001

"""max number of highest peaks"""
NMAX: int = 16

"""number of partials = 16 considered in harmonic finding (experimental)"""
NPARTIAL: int = 17

"""lower frequency limit, i.e. A0. May not need to be adjusted."""
FREQUENCY_LIMIT = 26.5

"""
initial max frequency [Hz] displayed, can be adjusted via ctrl-n/m or alt-n/m
and is reset via ctrl-r to default
"""
FREQUENCY_MAX: int = 2000
FREQUENCY_WIDTH_MIN: int = 500
FREQUENCY_STEP: int = 500

"""debug flag"""
DEBUG: bool = False
