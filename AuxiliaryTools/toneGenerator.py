"""
https://stackoverflow.com/questions/8299303/generating-sine-wave-sound-in-python
"""
import pyaudio
import numpy as np

# range [0.0, 1.0]
volume = .3
# sampling rate, Hz, must be integer
fs: int = 44100
# in seconds, may be float
duration: float = 5.
# sine frequency, Hz, may be float
f1: float = 415.0
# inharmonicity
inharmonicity: float = 1.e-4
# number of partitials
partials: int = 20

p = pyaudio.PyAudio()
f0 = f1 / np.sqrt(1. + inharmonicity)
alls = list()

l = []
for partial in range(1, partials + 1):
    f = f0 * partial * np.sqrt(1. + inharmonicity * partial ** 2)
    l.append(f)
    alls.append(
        # reduce volume of higher partials
        (np.cos(2. * np.pi * np.arange(fs * duration) * f / fs)) / partial**2
    )
print(l)
fade = np.exp(-np.arange(fs * duration) * 1. / fs)
samples = np.sum(alls, axis=0) * fade
max_volume = np.max(samples)
# generate samples, note conversion to float32 array
samples = (samples / max_volume * volume).astype(np.float32).tobytes()
# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

# play. May repeat with different volume values (if done interactively)
stream.write(samples)

stream.stop_stream()
stream.close()

p.terminate()
