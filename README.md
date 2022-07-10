# Harpsichord & Piano Tuning

### Introduction
An automatic tuning tool for string instruments, such as harpsichords and 
pianos.

The current application collects a mono audio signal from the computer's
input stream,
splits it into smaller, overlapping slices, and applies the Fourier transform to
each (known as short time Fourier transform: STFT). The slices have sizes of
2<sup>N</sup> samples, where N=15 or 16 (sampling period = 0.74 or 1.49 s, 
respectively), which overlap by multiples of 1024 samples. Each slice is 
apodized utilizing Hamming or Hann windowing, the full widths of their main 
lobes being 1.81 and 2.0 bins @ -6dB.

Adopting a reduction in resolution of 1.81 with 
Hamming, the resulting frequency resolution becomes 2.4 and 1,2 Hz for 
a sampling period = 0.74 and 1.49 s, respectively. Subsequently, in the
frequency domain, Butterworth high-pass filtering is applied to suppress noise
at the bottom, before the fundamental and overtone frequencies are sought.
In order to achieve higher accuracy in their positions, we fit Gaussians to the
NMAX strongest peaks found, with 5 data points on either side of the peak.
Their centroids are utilized in the derivation 
of the fundamental frequency and the inharmonicity factor.

For an ideal string the frequencies of higher partials are just multiples
of the fundamental frequency

**(1) <em>f<sub>n</sub> = n * f<sub>1</sub> </em>**, 

where n is the n<em>th</em> partial. 
The ear hears the fundamental frequency most prominently, 
but the overall sound is also colored by the presence of various overtones 
(frequencies greater than the fundamental frequency).
However, a real string behaves closer to a stiff bar according to a forth-order 
differential equation 
<img src="https://render.githubusercontent.com/render/math?math=\ddot y \propto {-y}\!\!''''">
with a quadratic dispersion. Its partials can be approximated by

**(2) <em>f<sub>n</sub> = n * f<sub>0</sub> * sqrt(1 + B * n<sup>2</sup>)</em>**

B and f<sub>0</sub> being the inharmonicity coefficient and base frequency, 
respectively. 

All peak positions are correlated to each other, such that they 
can be identifed as higher partials to one common base frequency f<sub>0</sub>. 
By rewriting (2), we get for two frequencies from all peak permutations

**(3) <em>B = (C - 1) / (j<sup>2</sup> - C * i<sup>2</sup>)</em>**, 
where 
**<em>C = (f<sub>j</sub> * i / f<sub>i</sub> * j) <sup>2</sup></em>**

**(4) <em>f<sub>0</sub> = f<sub>i</sub> / (i * sqrt(1 + B * i<sup>2</sup>))</em>**

The measured frequencies of the partials are denoted
**<em>f<sub>i</sub> < f<sub>j</sub></em>** and **1 &#8804;
<em>i < j &#8804; NPARTIAL</em>**. 
The maximum inharmonicity coefficient needs to be adjusted in
[parameters.py](https://github.com/Tamburasca/HarpsichordTuning/blob/master/Tuning/parameters.py), 
depending on the instrument to be tuned, B < 0.001 and < 0.05 for 
harpsichords and pianos, respectively.

### Features

The frequency of the first partial f<sub>1</sub> is 
compared to a value derived from the pitch level and a tuning table 
[tuningTable.py](https://github.com/Tamburasca/HarpsichordTuning/blob/master/Tuning/tuningTable.py), 
currently comprising Werkmeister III, 1/4 Comma Meantone, and Equal Temperament 
(feel free to edit/enhance it for yourself). We have not yet considered 
enharmonic equivalency in meantone, hence, one would have to enable/disable 
certain keys, 
such as A♭ vs. G#. The key in the center of the pie shows what key was 
pressed and its deviation, in units of cent, for the specified tuning and 
pitch level: too low (in red) and too high (in green).

![image info](./pictures/screenshot.png)

The orange vertical bars indicate the peaks identified by the peak 
finding routine. The red vertical bars show the partials up to 
<em>n<sub>max</sub> &#8804; NPARTIAL</em> as 
derived from the computed base frequency and inharmonicity coefficient 
when applying (2).

The hotkey 'ctrl-y' or 'x' stops the program or toggles between halt and 
resume, respectively. 'Ctrl-j' and 'ctrl-k' shorten and lengthen the shift 
between the audio slices, whereas 'ctrl-n' ('alt-n') and 'ctrl-m' ('alt-m') 
diminish and increase the max (min) frequency displayed. 'ctrl-r' resets 
parameter to initial values.
 
Run the program with: <em>python3 -m Tuning</em>

### Caveat

1) When tuning you may consider preventing the display from blanking, locking 
and the monitor's DPMS (on UNIX) energy saver from kicking in. To date I haven't 
found a decent solution yet that works for all OS flavors equally well. 
Suggestions welcome.

2) On certain Linux distributions, a package named python-tk (or similar) needs 
to be installed, when running in virtual environments.

3) Also note that the module pynput utilized here may encounter 
[plattform limitations](https://pynput.readthedocs.io/en/latest/limitations.html#)

### Results

We received, in preliminary results, accuracies of about 
0.21Hz@415Hz, which translates to about 0.09 cent and a reletive error of 
about 4x10<sup>-4</sup> in the inharmonicity factor. Stay tuned for detailed 
results.

### References

1) HARVEY FLETCHER, THE JOURNAL OF THE ACOUSTICAL SOCIETY OF AMERICA VOLUME 36,
NUMBER 1 JANUARY 1964
2) HAYE HINRICHSEN, REVISTA BRASILEIRA DE ENSINA FISICA, VOLUME 34, NUMBER 2,
2301 (2012)
3) JOONAS TUOVINEN, SIGNAL PROCESSING IN A SEMI-AUTOMATIC PIANO TUNING SYSTEM
(MA OF SCIENCE), AALTO UNIVERSITY, SCHOOL OF ELECTRICAL ENGINEERING (2019)

#### Contact

Ralf Antonius Timmermann, Email: rtimmermann@astro.uni-bonn.de, 
Argelander Institute for Astronomy (AIfA), University Bonn, Germany