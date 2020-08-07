# Harpsichord Tuning

An automatic tuning tool for string instruments, such as harpsichords and pianos. First experiments are published here. 
We still need to test with pianos, owing to the larger inharmonicity factor.

Collects a mono audio signal from a microphone in the input stream. The audio signal runs through a FFT with a
Hanning apodization. Subsequently, in the frequency domain, Gauss smoothing and Butterworth high-pass filtering is
 applied. The resulting spectrum is cross-correlated with a theoretical overtone spectrum for strings. The result of
 the cross-correlation is maximized. Inharmonicity of strings is considered here through equation

<img src="https://latex.codecogs.com/svg.latex?\Large&space;f_n = n * f_0 * \sqrt{1 + B * n^2}"/>

\
where n = 1, 2, 3, ... B is the inharmonicity coefficient and f<sub>0</sub> the fundamental frequency, respectively. 

The first partial f<sub>1</sub> is compared with a tuning table, currently comprising solely Werkmeister III, 
1/4 Comma Meantone, and Equal Temperament (feel free to edit/enhance for yourself). The deviation in units of cent is 
shown in the frequency plot, too low (in red), too high (in green).

See also:

1) HARVEY FLETCHER, THE JOURNAL OF THE ACOUSTICAL SOCIETY OF AMERICA VOLUME 36,
NUMBER 1 JANUARY 1964
2) HAYE HINRICHSEN, REVISTA BRASILEIRA DE ENSINA FISICA, VOLUME 34, NUMBER 2,
2301 (2012)
3) Joonas Tuovinen, Signal Processing in a Semi-AutomaticPiano Tuning System
(MA of Science), Aalto University, School of Electrical Engineering


Due to the low resolution of about 0.5 Hz/channel, the determination of inharmonicity coefficient seems somewhat 
difficult.
In order to achieve higher resolution the sampling interval needs to be enlarged, e.g. by pressing ctrl-k. However, that 
hampers the usability of this tool. Owing to the higher partials measured with this code, a determination of the 
inharmonicity coefficient appears feasible, though.

The hotkeys ctrl-y and ctrl-x exits and stops the program, respectively, ESC to resume. ctrl-j and ctrl-k shorten
and lengthen the recording interval within small ranges, whereas ctrl-n and ctrl-m diminish and increase the maximum 
frequency displayed in the lower frequency plot.

On UNIX OS please consider to run the package with sudo rights: <em>sudo python3 -m Tuning</em>, owing to
 the requirement of the keyboard module.