"""audio sampling rate in kHz derived from driver -> hardware & sound
do not modify!"""
RATE: int = 44100

"""size of slices: samples per FFT. Can be modifed to 2**N, where
N = 15/16 => 32768/65536 samples/slice: 0.743/1.486 sec sampling time/slice"""
SLICE_LENGTH: int = 2 ** 15

"""no of samples (initial value) by which each slice is shifted with regard 
to its previous, can be adjusted by +/-1024 samples through hotkeys crtl-j/k"""
SLICE_SHIFT: int = 16384

"""F_FILT: high pass cutoff frequency [Hz]"""
F_FILT: float = 200

"""F_ORDER: order of high pass Butterworth filter"""
F_ORDER: int = 2

"""noise level needs to be adjusted such, that there are no peaks detected with 
no key pressed (silence). This needs to be worked on in a later version."""
NOISE_LEVEL: float = 75.

"""minimum distance between two peaks in channels to identify a peak as such"""
DISTANCE: int = 8

"""(min, max) width of peak in channels to identify a peak as such"""
WIDTH: tuple = (0, 8)

"""window for Gauss fits: no of channels on either side"""
FIT_WINDOW: int = 5

"""max. inharmonicity of strings considered (harpsichord, piano, ...)"""
INHARM: float = 0.001

"""max number of highest peaks"""
NMAX: int = 16

"""number of partials considered in harmonic finding"""
NPARTIAL: int = 11

"""initial max frequency [Hz] displayed, can be adjusted through ctrl-n/m"""
FREQUENCY_MAX: int = 2000
FREQUENCY_WIDTH_MIN: int = 500
FREQUENCY_STEP: int = 500

"""debug flag"""
DEBUG: bool = False

"""logging format"""
myformat = "%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s"
