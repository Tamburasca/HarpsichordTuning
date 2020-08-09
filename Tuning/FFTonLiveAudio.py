#!/usr/bin/env python

"""
FFTonLiveAudio Copyright (C) 2020 Dr. Ralf Antonius Timmermann

A graphical tuning device for string instruments, such as (1) harpsichords
and (2) pianos. Still testing!

Collects an audio signal from an input stream, that runs through a FFT with
a Hanning apodization. Subsequently, in the frequency domain we apply Gauss
smoothing and a Butterworth high-pass filter, which is followed by a peak
finding. Of those peaks we try to find the partials (overtones). The
fundamental is compared with a tuning table (feel free to enhance for
yourself). The deviation in units of cent is shown in the frequency plot,
too low (in red), too high (in green).

Inharmonicity of strings is considered by the equation
f_n = n * f_1 * sqrt(1 + B * n**2), where n = 1, 2, 3, ... and
B is the inharmonicity coefficient. The maximum inharmonicity accepted is
defined in parameters.py Change accordingly for harpsichords and pianos.

see also
1) HARVEY FLETCHER, THE JOURNAL OF THE ACOUSTICAL SOCIETY OF AMERICA VOLUME 36,
NUMBER 1 JANUARY 1964
2) HAYE HINRICHSEN, REVISTA BRASILEIRA DE ENSINA FISICA, VOLUME 34, NUMBER 2,
2301 (2012)
3) Joonas Tuovinen, Signal Processing in a Semi-AutomaticPiano Tuning System
(MA of Science), Aalto University, School of Electrical Engineering

The hotkeys ctrl-y and ctrl-x exits and stops the program, respectively,
ESC to resume. Ctrl-j and ctrl-k shorten and lengthen the recording interval,
whereas ctrl-n and ctrl-m diminish and increase the max frequency displayed.

This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it under
certain conditions.
"""

__author__ = "Dr. Ralf Antonius Timmermann"
__copyright__ = "Copyright (C) Dr. Ralf Antonius Timmermann"
__credits__ = ""
__license__ = "GPLv3"
__version__ = "0.2"
__maintainer__ = "Dr. Ralf A. Timmermann"
__email__ = "rtimmermann@astro.uni-bonn.de"
__status__ = "Development"

print(__doc__)

import pyaudio
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import timeit
import time
import keyboard
from .multiProcess_opt import threaded_opt
from .tuningTable import tuningtable
from .parameters import _debug, INHARM


# do not modify below
FORMAT = pyaudio.paInt16
# mono, not stereo
CHANNELS = 1
# rate 48 kHz derived from driver -> hardware & sound
RATE = 44100
SIGMA = 1


class Tuner:

    def __init__(self, tuning, a1):
        """
        :param a1: float
            tuning frequency for a1
        :param tuning: string
            tuning temperament
        """
        self.record_seconds = 2.  # lengths of audio signal chunks, can be adjusted
        self.fmax = 2000.  # maximum frequency display in 2nd subplot, can be adjusted
        self.a1 = a1
        self.tuning = tuning  # see tuningTable.py
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
        _firstplot = True
        plt.ion()  # Stop matplotlib windows from blocking

        # start Recording
        self.stream.start_stream()

        while self.stream.is_active():

            print('Started Audio Stream ...')
            _start = timeit.default_timer()

            time.sleep(self.record_seconds)
            self.stream.stop_stream()  # stop the input stream for the time being
            print('Stopped Audio Stream ... Analyzing')

            # Convert the list of numpy-arrays into a 1D array (column-wise)
            amp = np.hstack(self.callback_output)
            # clear input stream
            self.callback_output = []

            _stop = timeit.default_timer()
            if _debug:
                print("time utilized for Audio [s]: " + str(_stop - _start))

            # hotkey interrupts
            if self.rc == 'x':
                keyboard.wait('esc')
                self.rc = None
            elif self.rc == 'y':
                return 'y'

            samples = len(amp)
            print('Number of samples', samples)
            t = np.arange(samples) / RATE
            resolution = RATE / samples
            print('Resolution (Hz/channel): ', resolution)

            t1, yfft = self.fft(amp=amp,
                                samples=samples)  # calculate FFT

            peakPos, peakHeight = self.peak(spectrum=yfft)  # peakfinding
            peakList = t1[peakPos]

            if peakList is not None:
                f_measured = self.harmonics(amp=yfft,
                                            freq=t1,
                                            ind=peakList,
                                            height=peakHeight)  # find the key
            else:
                f_measured = []

            displayed_text = ""
            color = 'none'
            if f_measured.size != 0:  # if key is found print it and its offset colored red or green
                tone, displaced = self.find(f_measured=f_measured[0])
                if tone:
                    displayed_text = "{2:s} (a1={3:3.0f}Hz) {0:s} offset={1:.0f} cent"\
                        .format(tone, displaced, self.tuning, self.a1)
                    color = 'green' if displaced >= 0 else 'red'

            _start = timeit.default_timer()

            if _firstplot:  # instantiate first plot and copy background
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
            else:
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

            for c in ax1.collections:
                c.remove()
            yevents = EventCollection(positions=f_measured,
                                      color='tab:red',
                                      linelength=0.05*np.max(yfft),
                                      linewidth=2.)
            ax1.add_collection(yevents)

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

            # resume the audio streaming, expect some retardation for the status change
            self.stream.start_stream()

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
        NMAX = 5
        list1 = []
        list2 = []
        _start = timeit.default_timer()

        # find peak according to prominence and remove peaks below threshold
        # prominences = signal.peak_prominences(x=spectrum, peaks=peaks)[0]
        # spectrum[peaks] == prominences with zero baseline
        peaks, _ = signal.find_peaks(x=spectrum,
                                     distance=16.,
                                     prominence=np.max(spectrum)/50.,  # sensitivity minus background
                                     width=(0, 20))  # max peak width -> needs adjustment!!!
        nPeaks = len(peaks)
        # consider NMAX highest, sort key = amplitude descending
        if nPeaks != 0:
            list1, list2 = (list(t) for t in zip(*sorted(zip(spectrum[peaks], peaks), reverse=True)))
            del list2[NMAX:]
            del list1[NMAX:]
        # re-sort again with sort key = frequency ascending
            list2, list1 = (list(t) for t in zip(*sorted(zip(list2, list1))))

        _stop = timeit.default_timer()
        if _debug:
            print("Peaks found:", nPeaks)
            print("Time for peak finding [s]:", _stop - _start)

        return list2, list1

    def fft(self, amp, samples=None):
        """
        performs FFT on a Hanning apodized time series and a Gauss smoothing afterwards. High pass filter performed as
        well.

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
        filt = signal.gaussian(M=21,
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

    def harmonics(self, amp, freq, ind, height=None):
        """
        :param amp: ndarray
            amplitudes of FFT transformed spectrum
        :param freq: ndarray
            freqencies of FFT transformed spectrum (same dimension of amp
        :param ind: ndarray
            peak positions found in FFT spectrum (NMAX highest)
        :param height: ndarray
            peak height found in FFT spectrum (NMAX highest)
        :return:
        ndarray
            positions of first 8 partials as found in fit
        """
        Npartial = 11
        initial = []

        _start = timeit.default_timer()

        if _debug:
            print(ind)
        if len(ind) > 1:
            # Median - Adjustive Trajectories (MAT)
            # Loop through all the peak positions and evaluate f_0 and b for each combination
            for i_ind in ind[:-1]:
                for j_ind in ind[1:]:
                    try:
                        # Loop through the partials up to Npartial
                        for m in range(1, Npartial-1):
                            for k in range(m+1, Npartial):
                                tmp = (j_ind * m / k) ** 2
                                b = (tmp - i_ind ** 2) / ((k * i_ind) ** 2 - tmp * m ** 2)
                                # INHARM is also used in boundaries in multiProcess.py
                                if 0 <= b < INHARM:
                                    f_fundamental = i_ind / (m * np.sqrt(1. + b * m ** 2))
                                    if _debug:
                                        print("partial: {0:d} {1:d} lower: {2:7.2f} upper: {3:7.2f} "
                                              "b: {4:0.6f} fundamental: {5:7.2f}".
                                              format(m, k, i_ind, j_ind, b, f_fundamental))
                                    # pump it all to the minimizer and let him decide, what's best
                                    initial.append(tuple((f_fundamental, b)))
                                    raise StopIteration  # break two loops here
                    except StopIteration:
                        pass
        # it's only one element to optimize with
        elif len(ind) == 1:
            initial.append(tuple((ind[0], 0.)))

        opt = threaded_opt(amp, freq, initial)
        opt.run
        print("The best result is [f0 , B] = ", opt.best_x)

        # prepare for displaying vertical bars, and key finding etc.
        f_n = np.array([])
        if opt.best_x is not None:
            for n in range(1, 11):
                f_n = np.append(f_n, opt.best_x[0] * n * np.sqrt(1. + opt.best_x[1] * n**2))

        _stop = timeit.default_timer()
        if _debug:
            print("time utilized for minimizer [s]: " + str(_stop - _start))

        return f_n

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
            if self.record_seconds < 0.1:
                self.record_seconds = 0.1
            print("Recording Time: {0:1.1f}s".format(self.record_seconds))
        elif key == 'n':
            self.fmax -= 100
            if self.fmax < 500:
                self.fmax = 500
            print("Max frequency displayed: {0:1.0f}Hz".format(self.fmax))
        elif key == 'm':
            self.fmax += 100
            if self.fmax > 10000:
                self.fmax = 10000
            print("Max frequency displayed: {0:1.0f}Hz".format(self.fmax))

        return None


def main():

    for tune in tuningtable.keys():
        print("Tuning ({1:d}) {0:s}".format(tune, list(tuningtable).index(tune)))
    a = Tuner(tuning=list(tuningtable.keys())[int(input("Tuning Number?: "))],
              a1=float(input("base frequency for a1 in Hz?: ")))
    keyboard.add_hotkey('ctrl+x', a.on_press, args='x')
    keyboard.add_hotkey('ctrl+y', a.on_press, args='y')
    keyboard.add_hotkey('ctrl+j', a.on_press, args='j')
    keyboard.add_hotkey('ctrl+k', a.on_press, args='k')
    keyboard.add_hotkey('ctrl+n', a.on_press, args='n')
    keyboard.add_hotkey('ctrl+m', a.on_press, args='m')

    a.animate
    plt.close('all')

    return 0