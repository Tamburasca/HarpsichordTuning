# Harpsichord Tuning

A tuning device for string instruments, such as harpsichords and pianos. First experiments published here. 

Collects a mono audio signal from an input stream. The audio signal runs through a FFT with a Hanning apodization. On the frequency domain we subsequently perform a Gauss smoothing
and a Butterworth high-pass filter.
Thereafter, we run a peak finding of which only the highest peaks are considered. Of those peaks we try to find
their partials (overtones). The peak with the lowest frequency and at least 5 partials is selected and compared with a
tuning table, currently Werkmeister III, 1/4 Comma Meantone, and Equal Temperament (feel free to enhance for yourself). The deviation in units of cent is shown in the frequency plot,
too low (in red), too high (in green).

Inharmonicity of strings is considered here

f<sub>n</sub> = n * f<sub>1</sub> * sqrt(1 + B * n<sup>2</sup>)

where n = 1, 2, 3, ... and B is the inharmonicity coefficient.

See also:

HARVEY FLETCHER, THE JOURNAL OF THE ACOUSTICAL SOCIETY OF AMERICA VOLUME 36, NUMBER 1 JANUARY 1964

HAYE HINRICHSEN, REVISTA BRASILEIRA DE ENSINA FISICA, VOLUME 34, NUMBER 2, 2301 (2012)

Due to the low resolution of about .5 Hz/channel, the determination of B is difficult, particularly in the bass area. In
In order to achieve higher resolution the sampling interval would have to be to large.

The hotkeys ctrl-y and ctrl-x exits and stops the program, respectively, ESC to resume. ctrl-j and ctrl-k shorten
and lengthen the recording interval, whereas ctrl-n and ctrl-m diminish and increase the max frequency displayed.

On UNIX OS please consider to run the package with sudo rights: sudo python3 -m Tuning, owing to the keyboard module.
