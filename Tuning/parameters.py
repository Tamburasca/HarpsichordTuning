"""audio sampling rate in kHz derived from driver -> hardware & sound
do not modify!"""
RATE: int = 44100

"""size of slices: samples per FFT. Can be modifed to 2**N, where N is
32768: N = 15 => 0.743 sec sampling time/slice
65536: N = 16 => 1.486 sec sampling time/slice"""
SLICE_LENGTH: int = 32768  # 65536

"""no of samples (initial value) by which each slice is shifted with regard 
to its previous, can be adjusted by +/-1024 samples through hotkeys crtl-j/k"""
SLICE_SHIFT: int = 16384  # 8192

"""initial max frequency [Hz] displayed, can be adjusted through ctrl-n/m"""
FREQUENCY_MAX: float = 2000.

"""noise level needs to be adjusted such, that there are no peaks detected with 
no key pressed (silence). This needs to be worked on in a later version."""
NOISE_LEVEL: float = 100.

"""minimum distance between two peaks to identify a peak"""
DISTANCE: int = 6

"""min and max width of peak to identify a peak"""
WIDTH: tuple = (0, 8)

"""window for Gauss fits: no of channels on either side"""
FIT_WINDOW: int = 6

"""max. inharmonicity of strings considered (harpsichord, piano, ...)"""
INHARM: float = 0.001

"""max number of highest peaks"""
NMAX: int = 15

"""high pass cutoff frequency [Hz] @-3db"""
F_FILT: float = 120.

"""high pass Butterworth filter of order F_ORDER"""
F_ORDER: int = 2

"""number of partials considered in harmonic finding"""
NPARTIAL: int = 11

"""debug flag"""
DEBUG: bool = True

"""logging format"""
myformat = "%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s"
