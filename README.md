# Harpsichord Tuning

An automatic tuning tool for string instruments, mainly harpsichords and 
pianos. Tests are still needed with pianos, owing to their larger inharmonicity 
factor.

The current application collects a mono audio signal from the input stream,
splits it into smaller, overlapping slices, and applies the Fourier transform to
each (known as short time Fourier transform: STFT). The slices have a size of
2<sup>N</sup> samples, with N=15 or 16, and overlap by multiples of 1024
samples. Each slice is apodized with a Hanning window. Subsequently, in the
frequency domain, Butterworth high-pass filtering is applied to suppress noise
at low frequencies before the fundamental and overtone frequencies are sought.
In order to achieve higher accuracy in their positions, we fit a Gaussian to the
NMAX strongest peaks found. Their centroids are utilized in the derivation 
of the fundamental frequency and the inharmonicity factor.

For an ideal string the frequencies of higher partials are just multiples
of the fundamental frequency

**(1) <em>f<sub>n</sub> = n * f<sub>1</sub> </em>**, 

where n = 1, 2, 3, ... The ear hears the fundamental frequency most prominently, 
but the overall sound is also colored by the presence of various overtones 
(frequencies greater than the fundamental frequency).

However, a real string behaves more like a stiff bar. Its partials can be 
approximated by

**(2) <em>f<sub>n</sub> = n * f<sub>0</sub> * sqrt(1 + B * n<sup>2</sup>)</em>**

B and f<sub>0</sub> being the inharmonicity coefficient and base frequency, 
respectively. 

All peak positions are correlated to each other, such that they 
can be identifed as higher partials to one common base frequency f<sub>0</sub>. 
By rewriting (2), we get for two frequencies from all peak permutations

**(3) <em>B = (C - 1) (n<sub>j</sub><sup>2</sup> - C * 
n<sub>i</sub><sup>2</sup>)</em>**, 

where 
**<em>C = (f<sub>j</sub> * n<sub>i</sub> / f<sub>i</sub> * n<sub>j</sub>)
<sup>2</sup></em>**

**(4) <em>f<sub>0</sub> = f<sub>i</sub> / (n<sub>i</sub> *
sqrt(1 + B * n<sub>i</sub><sup>2</sup>))</em>**

The measured frequencies and their partials are denoted
**<em>f<sub>i</sub> < f<sub>j</sub></em>** and **1 &#8804;
<em>n<sub>i</sub> < n<sub>j</sub> &#8804; NPARTIAL</em>**. 
The maximum inharmonicity coefficient needs to be adjusted in
[parameters.py](https://github.com/Tamburasca/HarpsichordTuning/blob/master/Tuning/parameters.py), 
depending on the instrument to be tuned, B < 0.001 and < 0.05 for 
harpsichords and pianos, respectively.

The frequency of the first partial f<sub>1</sub> (fundamental) is 
compared to a value derived from the pitch level and a tuning table 
[tuningTable.py](https://github.com/Tamburasca/HarpsichordTuning/blob/master/Tuning/tuningTable.py), 
currently comprising Werkmeister III, 
1/4 Comma Meantone, and Equal Temperament (feel free to edit/enhance it 
for yourself). We have not yet considered enharmonic equivalency, hence, one 
would have to enable/disable certain keys, such as A♭ vs. G#. 
The text in the plot shows the key's 
deviation, in units of cent, for the specified tuning and pitch level: 
too low (in red) and too high (in green).

![image info](./pictures/screenshot.png)

The orange vertical bars indicate the peaks identified by the peak 
finding routine. The red vertical bars show the partials up to 
<em>n<sub>max</sub> &#8804; NPARTIAL</em> as 
derived from the computed base frequency and inharmonicity coefficient 
when applying (2).

The hotkey ctrl-y and ctrl-x exits and stops the program, respectively, 
ESC to resume. ctrl-j and ctrl-k shorten and lengthen the slice shift, 
whereas ctrl-n (alt-n) and ctrl-m (alt-m) diminish and increase the maximum 
(minimum) frequency that is displayed.

On certain Linux distributions, a package named python-tk (or similar) needs 
to be installed, when running in virtual environments.

Also note that the module pynput utilized here may encounter 
[plattform limitations](https://pynput.readthedocs.io/en/latest/limitations.html#)

Run the program with: <em>python3 -m Tuning</em>

References:

1) HARVEY FLETCHER, THE JOURNAL OF THE ACOUSTICAL SOCIETY OF AMERICA VOLUME 36,
NUMBER 1 JANUARY 1964
2) HAYE HINRICHSEN, REVISTA BRASILEIRA DE ENSINA FISICA, VOLUME 34, NUMBER 2,
2301 (2012)
3) JOONAS TUOVINEN, SIGNAL PROCESSING IN A SEMI-AUTOMATIC PIANO TUNING SYSTEM
(MA OF SCIENCE), AALTO UNIVERSITY, SCHOOL OF ELECTRICAL ENGINEERING (2019)
