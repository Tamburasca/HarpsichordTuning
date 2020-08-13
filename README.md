# Harpsichord Tuning

An automatic tuning tool for string instruments, such as harpsichords and 
pianos. First experiments are published here. 
We still need to test with pianos, owing to the larger inharmonicity factor.

Collects a mono audio signal from a microphone in the input stream. The audio 
signal runs through a FFT with a Hanning apodization. Subsequently, 
in the frequency domain, Gauss smoothing and Butterworth high-pass filtering 
is applied. The resulting spectrum is cross-correlated with a theoretical 
overtone spectrum for strings. The result of the cross-correlation 
is maximized. Inharmonicity of strings is considered here through equation

<em>f<sub>n</sub> = n * f<sub>0</sub> * sqrt(1 + B * n<sup>2</sup>)</em>

where n = 1, 2, 3, ... B and f<sub>0</sub> is the inharmonicity coefficient and 
fundamental frequency, respectively. The maximum inharmonicity coefficient
needs to be adjusted, depending on the instrument to be tuned, B < 0.001 and 
 < 0.01 for harpsichords and pianos, respectively. 
 
The first partial f<sub>1</sub> is compared to a tuning table, 
currently comprising solely Werkmeister III, 
1/4 Comma Meantone, and Equal Temperament 
(feel free to edit/enhance for yourself). The deviation in units of cent is 
shown in the frequency plot, too low (in red), too high (in green).

![image info](./pictures/screenshot.png)

The upper plot represents the audio signal in the time domain, whereas the lower
represents the frequency domain. The red vertical bars show the partials for 
<em>n<sub>max</sub> = 10</em> as 
derived from the base frequency and inharmonicity coefficient after maximizing 
cross correlation. The text in the upper right corner shows the 
deviation for a specific tuning.

See also:

1) HARVEY FLETCHER, THE JOURNAL OF THE ACOUSTICAL SOCIETY OF AMERICA VOLUME 36,
NUMBER 1 JANUARY 1964
2) HAYE HINRICHSEN, REVISTA BRASILEIRA DE ENSINA FISICA, VOLUME 34, NUMBER 2,
2301 (2012)
3) JOONAS TUOVINEN, SIGNAL PROCESSING IN A SEMI-AUTOMATIC PIANO TUNING SYSTEM
(MA OF SCIENCE), AALTO UNIVERSITY, SCHOOL OF ELECTRICAL ENGINEERING (2019)

The hotkeys ctrl-y and ctrl-x exits and stops the program, respectively, 
ESC to resume. ctrl-j and ctrl-k shorten and lengthen the recording 
interval within small ranges, whereas ctrl-n and
 ctrl-m diminish and increase the maximum 
frequency displayed in the lower frequency plot.

On UNIX OS please consider running the package with sudo rights: 
<em>sudo python3 -m Tuning</em>, owing to
 the requirement of the keyboard module.