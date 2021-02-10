"""audio sampling rate in kHz derived from driver -> hardware & sound
no necessity to modify!"""
RATE: int = 44100

"""noise level needs to be adjusted such, that there are no peaks detected with 
no key pressed (silence). This needs to be worked on in a later version."""
NOISE_LEVEL: float = 150

"""max. inharmonicity of strings considered (harpsichord, piano, ...)"""
INHARM: float = 0.001

"""max number of highest peaks"""
NMAX: int = 15

"""high pass cutoff frequency @-3db"""
F_FILT: float = 300. # guess

"""high pass Butterworth filter of order F_ORDER"""
F_ORDER: int = 1

"""number of partials considered in harmonic finding"""
NPARTIAL = 11

"""debug flag"""
DEBUG: bool = False
