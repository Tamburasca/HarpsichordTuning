# Harpsichord Tuning

An automatic tuning tool for string instruments, such as harpsichords and 
pianos. Further Tests are needed with pianos, owing to the larger inharmonicity 
factor.

Collects a mono audio signal from the input stream. The signal 
runs through a FFT with a Hanning apodization in the time domain. 
Subsequently, in the frequency domain, Butterworth high-pass filtering 
is applied before peaks are sought by means of Python's signal 
module (peak_find). In order to achieve higher accuracy in their 
positions, the measured peaks are fitted to a Gauss curve. All peak positions 
are correlated to each other to identify them as partials to
one common fundamental frequency, where 

<em>f<sub>n</sub> = n * f<sub>0</sub> * sqrt(1 + B * n<sup>2</sup>)</em>

with n = 1, 2, 3, ... B and f<sub>0</sub> as the inharmonicity coefficient and 
fundamental frequency, respectively. 

The maximum inharmonicity coefficient needs to be adjusted, depending on 
the instrument to be tuned, B < 0.001 and < 0.05 for harpsichords and 
pianos, respectively (see parameters.py).
 
The frequency of the first partial f<sub>1</sub> is compared to a value 
computed from the pitch level (input value for a1) and a tuning table 
(input value), currently 
comprising solely Werkmeister III, 1/4 Comma Meantone, and Equal Temperament 
(feel free to edit/enhance it for yourself). The deviation in units of cent is 
shown in the frequency plot, too low (in red), too high (in green).

![image info](./pictures/screenshot.png)

The upper plot represents the audio signal in the time domain, whereas the lower
represents its Fourier analysis in the frequency domain. The olive 
vertical bars indicate the peaks that were identified and located by their 
frequencies. The red vertical bars show the partials for 
<em>n<sub>max</sub> &#8804; NPARTIAL</em> as 
derived from the base frequency and inharmonicity coefficient.
The text in the upper right corner shows the 
deviation of the key for a specific tuning and pitch level.

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

On certain Linux distributions, a package named python-tk (or similar) needs 
to be installed, when running in virtual environments.

Also note that the module pynput utilized here 
may encounter 
[plattform limitations](https://pynput.readthedocs.io/en/latest/limitations.html#)

Run the program with: <em>python3 -m Tuning</em>