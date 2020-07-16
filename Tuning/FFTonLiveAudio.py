#!/usr/bin/env python

"""
A tuning device for string instruments, such as (1) harpsichords and (2) pianos.

Collects a mono audio signal from a input stream of length self.record_seconds and a sample rate RATE. The audio signal
runs through a subsequent FFT with a Hanning apodization. On the frequency domain we perform a Gauss smoothing
(currently ensabled) and a 4th order Butterworth high-pass filter (cutoff frequency F_FILT and order F_ORDER).
Hereafter, we run a peak finding of which only the NMAX highest peaks are considered. Of those NMAX peaks we try to find
its partials (overtones). The peak with the lowest frequency and at least 5 partials is selected and compared with a
tuning table (feel free to enhance for yourself). The deviation in units of cent is shown in the frequency plot,
too low (in red), too high (in green).

Inharmonicity of strings is considered here
f_n = f_1 * n * sqrt(1 + B*n**2), where n = 1, 2, 3, ... and B is the inharmonicity coefficient (max displacement
0 < b < 0.005 for a harpsichord)
see also
HARVEY FLETCHER, THE JOURNAL OF THE ACOUSTICAL SOCIETY OF AMERICA VOLUME 36, NUMBER 1 JANUARY 1964
HAYE HINRICHSEN, REVISTA BRASILEIRA DE ENSINA FISICA, VOLUME 34, NUMBER 2, 2301 (2012)

Due to the low resolution, particularly in the bass area, the determination of B is difficult.

The hotkeys ctrl-y and ctrl-x exits and stops the program, ESC to resume. ctrl-j and ctrl-k shorten and lengthen the
recording interval, wherease ctrl-n and ctrl-m diminish and increase the max displayed frequency.

Runs with
python      3.8
keyboard    0.13.5
PyAudio     0.2.11
numpy       >1.18.1
scipy       >1.4.1
matplotlib  >3.2.1
"""

__author__ = "Dr. Ralf Antonius Timmermann"
__copyright__ = "Copyright (C) Dr. Ralf Antonius Timmermann"
__credits__ = ""
__license__ = "GPLv3"
__version__ = "0.1"
__maintainer__ = "Dr. Ralf A. Timmermann"
__email__ = "ralf.timmermann@gmx.com"
__status__ = "Development"

import pyaudio
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import timeit
import time
import keyboard


# do not modify below
FORMAT = pyaudio.paInt16
# mono, not stereo
CHANNELS = 1
# rate 48 kHz derived from driver -> hardware & sound
RATE = 44100
SIGMA = 2
#
_debug = False

tuningtable = {
    'Werkmeister I(III)': {
        'C': 0,
        'C#': 90,
        'D': 192,
        'D#': 294,
        'E': 390,
        'F': 498,
        'F#': 588,
        'G': 696,
        'G#': 792,
        'A': 888,
        'B♭': 996,
        'B': 1092
        },
    '1/4 Comma Meantone': {
        'C': 0,
        'C#': 76.05,
        'D': 193.16,
        # enable/disable to your tuning preference
        # 'D#': 269.21,
        'E♭': 310.26,
        'E': 386.31,
        'F': 503.42,
        'F#': 579.47,
        'G': 696.58,
        # 'A♭': 813.69,
        'G#': 772.63,
        'A': 889.74,
        'B♭': 1006.84,
        'B': 1082.89
        },
    'Equal Temperament': {
        'C': 0,
        'C#': 100,
        'D': 200,
        'D#': 300,
        'E': 400,
        'F': 500,
        'F#': 600,
        'G': 700,
        'G#': 800,
        'A': 900,
        'B': 1000,
        'H': 1100
        }
}


class Tuner:

    def __init__(self, tuning, a1):
        """
        :param a1: float
            tuning frequency for a1
        :param tuning: string
            tuning temperament
        """
        # lengths of audio signal chunks
        self.record_seconds = 2.
        self.fmax = 2000.
        self.a1 = a1
        self.tuning = tuning
        self.rc = None

        self.callback_output = []
        audio = pyaudio.PyAudio()
        self.stream = audio.open(format=FORMAT,
                                 channels=CHANNELS,
                                 rate=RATE,
                                 output=False,
                                 input=True,
                                 stream_callback=self.callback)

    def callback(self, in_data, frame_count, time_info, flag):
        """
        :param in_data:
        :param frame_count:
        :param time_info:
        :param flag:
        :return:
        """
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.callback_output.append(audio_data)

        return None, pyaudio.paContinue

    @property
    def animate(self):
        """
        calling routine for audio, FFT, peak and partials and key finding, and plotting. Listens for events in plot
        window

        :return:
        string
            return code
        """
        # upper frequency limit for plots
        _firstplot = True
        plt.ion()  # Stop matplotlib windows from blocking

        # start Recording
        self.stream.start_stream()
        while self.stream.is_active():

            print('Starting Audio Stream...')
            _start = timeit.default_timer()

            time.sleep(self.record_seconds)
            # Convert the list of numpy-arrays into a 1D array (column-wise)
            amp = np.hstack(self.callback_output)
            # clear stream
            self.callback_output = []

            _stop = timeit.default_timer()
            if _debug:
                print("time utilized for Audio [s]: " + str(_stop - _start))

            # hotkey interrupts
            if self.rc == 'x':
                keyboard.wait('esc')
                self.rc = None
                return 'x'
            elif self.rc == 'y':
                return 'y'

            samples = len(amp)
            print('Number of samples', samples)
            t = np.arange(samples) / RATE
            resolution = RATE / samples
            print('Resolution (Hz/channel): ', resolution)

            # calculate FFT
            t1, yfft = self.fft(amp=amp,
                                samples=samples)
            # peakfinding
            peakPos, peakHeight = self.peak(spectrum=yfft)
            peakList = t1[peakPos]
            # find the key
            f_measured = self.harmonics(ind=peakList,
                                        height=peakHeight)
            # if key found print it and its offset colored red or green
            if f_measured:
                tone, displaced = self.find(f_measured=f_measured)
                if tone:
                    displayed_text = "{2:s} (a1={3:3.0f}Hz) {0:s} offset={1:.0f} cent"\
                        .format(tone, displaced, self.tuning, self.a1)
                    color = 'green' if displaced >= 0 else 'red'
            else:
                displayed_text = ""
                color = 'none'

            # instantiate first plot and copy background
            if _firstplot:
                # Setup figure, axis, lines, text and initiate plot once and copy background
                fig = plt.gcf()
                ax = fig.add_subplot(211)
                ax1 = fig.add_subplot(212)
                fig.set_size_inches(12, 8)
                ln, = ax.plot(t, amp)
                ln1, = ax1.plot(t1, yfft)
                text = ax1.text(self.fmax, np.max(yfft), "",
                                # color='',
                                verticalalignment='top',
                                horizontalalignment='right',
                                fontsize=11,
                                fontweight='bold'
                                )
                ax.set_xlabel('Time/s')
                ax.set_ylabel('Intensity/arb. units')
                ax1.set_xlabel('Frequency/Hz')
                ax1.set_ylabel('Intensity/arb. units')
                axbackground = fig.canvas.copy_from_bbox(ax.bbox)
                ax1background = fig.canvas.copy_from_bbox(ax1.bbox)
            # upper subplot
            ln.set_xdata(t)
            ln.set_ydata(amp)
            ax.set_xlim([0., np.max(t)])
            # lower subplot
            ln1.set_xdata(t1)
            ln1.set_ydata(yfft)
            ax1.set_xlim([0., self.fmax])
            # set text attributes of lower subplot
            text.set_text(displayed_text)
            text.set_color(color)
            text.set_x(self.fmax)
            text.set_y(np.max(yfft))

            _start = timeit.default_timer()

            for c in ax1.collections:
                c.remove()
            xevents = EventCollection(positions=peakList,
                                      color='tab:orange',
                                      linelength=0.02*np.max(yfft))
            ax1.add_collection(xevents)
            # Rescale the axis so that the data can be seen in the plot
            # if you know the bounds of your data you could just set this once
            # so that the axis don't keep changing
            ax.relim()
            ax.autoscale_view()
            ax1.relim()
            ax1.autoscale_view()
            if _firstplot:
                fig.canvas.draw()
                _firstplot = False
            else:
                # restore background
                fig.canvas.restore_region(axbackground)
                fig.canvas.restore_region(ax1background)
                # redraw just the points
                ax.draw_artist(ln)
                ax1.draw_artist(ln1)
                ax1.draw_artist(text)
                # fill in the axes rectangle
                fig.canvas.blit(ax.bbox)
                fig.canvas.blit(ax1.bbox)
            fig.canvas.flush_events()

            _stop = timeit.default_timer()
            if _debug:
                print("time utilized for matplotlib [s]: " + str(_stop - _start))

        return self.rc

    def peak(self, spectrum):
        """
        find peaks in frequency spectrum
        :param spectrum: list
            spectrum from FFT
        :return:
        list1
            list of peak heights
        list2
            list of frequencies
        """
        # max number of highest peaks
        NMAX = 25
        list1 = []
        list2 = []
        _start = timeit.default_timer()

        # find peak according to prominence and remove peaks below threshold
        # prominences = signal.peak_prominences(x=spectrum, peaks=peaks)[0]
        # spectrum[peaks] == prominences with zero baseline
        peaks, _ = signal.find_peaks(x=spectrum,
                                     distance=8.,
                                     # sensitivity minus background
                                     prominence=np.max(spectrum)/100,
                                     # max peak width -> needs adjustment!!!
                                     width=(0, 20))
        nPeaks = len(peaks)
        print("Peaks found:", nPeaks)
        # consider NMAX highest, sort key = amplitude descending
        if nPeaks != 0:
            list1, list2 = (list(t) for t in zip(*sorted(zip(spectrum[peaks], peaks), reverse=True)))
            del list2[NMAX:]
            del list1[NMAX:]
            # re-sort sort key = frequency ascending
            list2, list1 = (list(t) for t in zip(*sorted(zip(list2, list1))))

        _stop = timeit.default_timer()
        if _debug:
            print("Time for peak finding [s]:", _stop - _start)

        return list2, list1

    def fft(self, amp, samples=None):
        """
        performs FFT on a Hanning apodized time series and a Gauss smoothing afterwards

        :param amp: list float
            time series
        :param samples: int optional
             number of samples
        :return:
        list float
            frequency values of spectrum
        list float
            intensities
        """
        F_FILT = 25  # high pass cutoff frequency @-3db
        F_ORDER = 4  # high pass Butterworth filter of order F_ORDER
        _start = timeit.default_timer()

        if not samples:
            samples = len(amp)
        hanning = np.hanning(samples)
        # gaussian = windows.gaussian(samples, std=0.1 * samples)
        y_raw = np.fft.rfft(hanning * amp)
        t1 = np.fft.rfftfreq(samples, 1. / RATE)
        y = np.abs(y_raw)
        # PDS
        # y = 2. * RATE * np.abs(y_raw) ** 2 / samples
        # convolve with a Gaussian of width SIGMA
        filt = signal.gaussian(M=31,
                               std=SIGMA)
        spectrum = signal.convolve(in1=y,
                                   in2=filt,
                                   mode='same')
        # high pass Butterworth filter
        b, a = signal.butter(F_ORDER, F_FILT, 'high', analog=True)
        _, h = signal.freqs(b, a, worN=t1)
        y_final = np.abs(h) * spectrum

        _stop = timeit.default_timer()
        if _debug:
            print("time utilized for FFT [s]: " + str(_stop - _start))

        return t1, y_final

    def find(self, f_measured):
        """
        finds key and its offset from true key for given a temperament

        :param f_measured: float
            measured frequency
        :return:
        string
            measured key
        float, None
            offset from true key in cent or None if error
        """
        _start = timeit.default_timer()
        for i in range(-3, 6):
            offset = np.log2(f_measured / self.a1) * 1200 - i * 1200
            for key, value in tuningtable[self.tuning].items():
                displaced = offset + tuningtable[self.tuning].get('A') - value
                if -60 < displaced < 60:
                    _stop = timeit.default_timer()
                    if _debug:
                        print(i, key, value, displaced + tuningtable[self.tuning].get('A') - value)
                        print("time utilized for Find [s]: " + str(_stop - _start))

                    return key, displaced

        return None, None

    def inharmonicity(self, f1, fn, k):
        """
        calculates the inharmonicity coefficient and fundamental frequency from its partials

        :param f1: float
            base frequency 1st partial
        :param fn: float
            frequency of partial
        :param k: int
            number of partial
        :return:
        float or None
            inharmonicity coefficient
        float on None
            fundamental frequency (calculatory)
        """
        # returns b and f_fundamental
        BMAX = 0.001
        partial_r = (fn / (k * f1)) ** 2
        if (k ** 2 - partial_r) > 0:
            b = (partial_r - 1.) / (k ** 2 - partial_r)
            # -rethink -BMAX to allow for the low resolution !!!!
            if -BMAX <= b <= BMAX:
                f_fundamental = f1 / np.sqrt(1. + b)

                return b, f_fundamental

        return None, None

    def harmonics(self, ind, height=None):
        """

        :param ind: list float
            list of peaks (sort key: frequency)
        :param height: list float (optional)
            list of peak heights (sort key: frequency)
        :return:
        float
            base frequency of lowest peak found with at least 5 partials
        """
        NPARTIALS = 5
        NN = 8
        _start = timeit.default_timer()

        # f_1 > f_fundamental
        for i in range(0, len(ind)-1):
            # f_n partial
            b_sum = 0.
            f_fundamental_sum = 0
            height_max = height[i]
            _counter_f1 = 1
            for j in range(1, len(ind)):
                # seek partials up to number 8
                for k in range(2, NN):  # k is partial of fundamental frequency
                    # b is the inharmonicity coefficient b < 0.005 for a harpsichord
                    b, f_fundamental = self.inharmonicity(ind[i], ind[j], k)
                    if b is not None:
                        print("partial: {0:d} lower: {1:8.2f}Hz upper: {2:8.2f}Hz counter: {3:d}"
                              " b: {4: 0.6f} fundamental: {5:8.2f}Hz"
                              .format(k, ind[i], ind[j], _counter_f1, b, f_fundamental))
                        b_sum += b
                        f_fundamental_sum += f_fundamental
                        _counter_f1 += 1
                        height_max = max(height_max, height[j])
            # avarage values for f0 and b to calculate first partial f1
            if _counter_f1 >= NPARTIALS:
                b_sum /= (_counter_f1-1)
                f_fundamental_sum /= (_counter_f1-1)
                f1 = f_fundamental_sum * np.sqrt(1. + b_sum)
                print("f1: {0:4.2f} Hz, no partials: {1}, inharm.coeff: {2:.5f}, fundamental freq.: {3:4.2f}"
                      .format(f1, _counter_f1-1, b_sum, f_fundamental_sum))

                _stop = timeit.default_timer()
                if _debug:
                    print("Time for partials finding [s]:", _stop - _start)

                return f1

        _stop = timeit.default_timer()
        if _debug:
            print("Time for partials finding [s]:", _stop - _start)

        return None

    def on_press(self, key):
        """
        interrupts on a key and set self.rc accordingly
        :param key: string
            interrupt key
        :return:
            None
        """
        if key == 'x':
            print("continue with ESC")
            self.stream.stop_stream()
            self.rc = 'x'
        elif key == 'y':
            print("quitting...")
            self.stream.stop_stream()
            self.rc = 'y'
        elif key == 'k':
            self.record_seconds += 0.1
            print("Recording Time: {0:1.1f}s".format(self.record_seconds))
        elif key == 'j':
            self.record_seconds -= 0.1
            if self.record_seconds < 0:
                self.record_seconds = 0
            print("Recording Time: {0:1.1f}s".format(self.record_seconds))
        elif key == 'n':
            self.fmax -= 100
            if self.fmax < 500:
                self.fmax = 500
        elif key == 'm':
            self.fmax += 100
            if self.fmax > 10000:
                self.fmax = 10000

        return None


def main():
    for tune in tuningtable.keys():
        print("Tuning ({1:d}) {0:s}".format(tune, list(tuningtable).index(tune)))
    a = Tuner(tuning=list(tuningtable.keys())[int(input("Tuning Number?: "))],
              a1=float(input("base frequency a1 in Hz?: ")))
    keyboard.add_hotkey('ctrl+x', a.on_press, args='x')
    keyboard.add_hotkey('ctrl+y', a.on_press, args='y')
    keyboard.add_hotkey('ctrl+j', a.on_press, args='j')
    keyboard.add_hotkey('ctrl+k', a.on_press, args='k')
    keyboard.add_hotkey('ctrl+n', a.on_press, args='n')
    keyboard.add_hotkey('ctrl+m', a.on_press, args='m')

    while a.animate == 'x':
        # restart
        plt.close('all')
