#!/usr/bin/env python

"""
FFTonLiveAudio Copyright (C) 2020-21, Dr. Ralf Antonius Timmermann

A graphical tuning tool for string instruments, such as (1) harpsichords
and (2) pianos.

Collects an audio signal from the input stream, that runs through a FFT.
In the frequency domain peak finding is applied. Of all peaks found, in all
combinations, they are identified as common partials (overtones) to one
fundamental. It is then compared to a tuning table (input value, feel free
to enhance special non-equal tunings yourself) and a given pitch value
(input value). The deviation in units of cent is shown, too low (red),
too high (green).

Inharmonicity of strings is considered through equation
f_n = n * f_1 * sqrt(1 + B * n**2), where n = 1, 2, 3, ... and
B is the inharmonicity coefficient. The maximum inharmonicity accepted is
defined in parameters.py Change accordingly for harpsichords and pianos.

References:
1) HARVEY FLETCHER, THE JOURNAL OF THE ACOUSTICAL SOCIETY OF AMERICA VOLUME 36,
NUMBER 1 JANUARY 1964
2) HAYE HINRICHSEN, REVISTA BRASILEIRA DE ENSINA FISICA, VOLUME 34, NUMBER 2,
2301 (2012)
3) Joonas Tuovinen, Signal Processing in a Semi-AutomaticPiano Tuning System
(MA of Science), Aalto University, School of Electrical Engineering (2019)

The hotkeys ctrl-y and ctrl-x exits and stops the program, respectively,
ESC to resume. Ctrl-j and ctrl-k shorten and lengthen the shift between the
individual audio slices, whereas ctrl-n and ctrl-m diminish and increase the
max frequency displayed. Parameters can be adjusted in the parameters.py file.

This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it under
certain conditions.
"""

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
from skimage import util
import timeit
import time
from pynput import keyboard
import logging
from operator import itemgetter
from .tuningTable import tuningtable
from .FFTroutines import fft, peak, harmonics
from Tuning import parameters

__author__ = "Dr. Ralf Antonius Timmermann"
__copyright__ = "Copyright (C) Dr. Ralf Antonius Timmermann"
__credits__ = ""
__license__ = "GPLv3"
__version__ = "2.0.0"
__maintainer__ = "Dr. Ralf A. Timmermann"
__email__ = "rtimmermann@astro.uni-bonn.de"
__status__ = "QA"

print(__doc__)

logging.basicConfig(format=parameters.myformat,
                    level=logging.INFO,
                    datefmt="%H:%M:%S")
if parameters.DEBUG:
    logging.getLogger().setLevel(logging.DEBUG)


class Tuner:
    def __init__(self, **kwargs):
        """
        :param a1: float
            pitch frequency for a1
        :param tuning: string
            tuning temperament
        """
        self.step = parameters.SLICE_SHIFT
        self.fmax = parameters.FREQUENCY_MAX
        self.a1 = kwargs.get('a1')
        self.tuning = kwargs.get('tuning')  # see tuningTable.py
        self.rc = None

        self.callback_output = list()
        audio = pyaudio.PyAudio()
        self.stream = audio.open(format=pyaudio.paInt16,
                                 channels=1,
                                 rate=parameters.RATE,
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

    def on_activate_x(self):
        # suspend the audio stram and freezes the plat
        print("continue with ESC or quit with ctrl-y")
        self.rc = 'x'

    def on_activate_y(self):
        # exits the program
        print("quitting...")
        self.rc = 'y'

    def on_activate_k(self):
        # increases the shift by which the slices progress
        self.step += 1024
        if self.step > parameters.SLICE_LENGTH:
            self.step = parameters.SLICE_LENGTH
        print("Slice shift: {0:d} bytes".format(self.step))

    def on_activate_j(self):
        # decreases the shift by which the slices progress
        self.step -= 1024
        if self.step < 4096:
            self.step = 4096
        print("Slice shift: {0:d} bytes".format(self.step))

    def on_activate_n(self):
        # decreases the max. frequency plotted
        self.fmax -= 500 if self.fmax > 2000 else 100
        if self.fmax < 500:
            self.fmax = 500
        print("Max frequency displayed: {0:1.0f}Hz".format(self.fmax))

    def on_activate_m(self):
        # increases the max. frequency plotted
        self.fmax += 500 if self.fmax >= 2000 else 100
        if self.fmax > 15000:
            self.fmax = 15000
        print("Max frequency displayed: {0:1.0f}Hz".format(self.fmax))

    def on_activate_esc(self):
        # resume the audio streaming
        self.rc = 'esc'

    def find(self, f_measured):
        """
        finds key and its offset from true key for given a temperament
        :param f_measured: float
            measured frequency
        :return:
        string
            measured key
        float, None
            offset from true key in cent or None if None found or error
        """

        def timeusage():
            _stop = timeit.default_timer()
            logging.debug("time utilized for key finding [s]: {0}".format(
                str(_stop - _start)))

        _start = timeit.default_timer()

        for i in range(-4, 5):  # key range from keys C0 till B8
            offset = np.log2(f_measured / self.a1) * 1200 - i * 1200
            for key, value in tuningtable[self.tuning].items():
                displaced = offset + tuningtable[self.tuning].get('A') - value
                if -60 < displaced < 60:
                    logging.debug("{0} {1} {2} {3}"
                                  .format(str(i),
                                          str(key),
                                          str(value),
                                          str(displaced +
                                              tuningtable[self.tuning].get('A')
                                              - value)))
                    timeusage()
                    return key, displaced

        timeusage()
        return None, None

    def slice(self):
        """
        collects the audio data and arranges it into slices of length
        SLICES_LENGTH shifted by SLICES_SHIFT
        :return:
        ndarray
            (rolling) window view of the input array. If arr_in is
            non-contiguous, a copy is made.
        """
        # interrupt on hotkey 'ctrl-x', resume on 'esc' and clear buffer
        if self.rc == 'x':
            self.stream.stop_stream()
            while self.rc not in ['esc', 'y']:
                # loop and wait until ESC or ctrl-y is pressed
                time.sleep(.1)
            if self.rc == 'esc':
                self.callback_output = list()
                self.stream.start_stream()
                logging.info(
                    "Clearing audio buffer and resuming audio stream ...")
                self.rc = None
            elif self.rc == 'y':
                return None
        # exit program on hotkey ctrl-y
        elif self.rc == 'y':
            self.stream.stop_stream()
            return None

        logging.debug("=== starting cycle ===")
        _start = timeit.default_timer()
        # wait until buffer filled by at least one FFT slice, where
        # length is in units of buffer = 1024
        while len(self.callback_output) < parameters.SLICE_LENGTH // 1024:
            time.sleep(0.001)
        # Convert the list of numpy-arrays into a 1D array (column-wise)
        amp = np.hstack(self.callback_output)
        slices = util.view_as_windows(amp,
                                      window_shape=(
                                          parameters.SLICE_LENGTH,),
                                      step=self.step)
        _stop = timeit.default_timer()
        logging.debug("Audio shape: {0}, Sliced audio shape: {1}"
                      .format(amp.shape,
                              slices.shape))
        logging.debug("time utilized for Audio [s]: {0}".format(
            str(_stop - _start)))

        return slices

    def animate(self):
        """
        calling routine for audio, FFT, peak and partials and key finding,
        and plotting. Listens for events in plot window
        :return:
        string
            return code
        """
        _firstplot = True
        plt.ion()  # Stop matplotlib windows from blocking

        resolution = parameters.RATE / parameters.SLICE_LENGTH * 1.5
        logging.debug(
            "Resolution incl. Hanning apodization (Hz/channel) ~ {0}"
            .format(str(resolution)))

        # start Recording
        self.stream.start_stream()

        # main loop while audio stream active
        while self.stream.is_active():
            slices = self.slice()
            if self.rc == 'y':
                return self.rc

            # work off all slices, before pulling from audio stream
            for sl in slices:
                logging.debug("no of slices: " + str(len(slices)))
                # remove first step of slice
                del self.callback_output[0:self.step // 1024]
                # calculate FFT
                t1, yfft = fft(amp=sl)
                # peakfinding
                peaks = peak(frequency=t1,
                             spectrum=yfft)
                peaklist = list(map(itemgetter(0), peaks))
                # harmonics
                if peaks is not None:
                    f_measured = harmonics(peaks=peaks)  # find the key

                displayed_text = ""
                color = None
                displayed_title = "{0:s} (a1={1:3.0f}Hz)".format(self.tuning,
                                                                 self.a1)
                info_text = "Resolution: {2:3.1f} Hz/channel\n" \
                            "Audio shape: {0} [slices, samples]\n" \
                            "Slice shift: {1:d} samples".format(slices.shape,
                                                                self.step,
                                                                resolution)
                info_color = 'red' if slices.shape[0] > 3 else 'black'
                font_title = {'family': 'serif',
                              'color': 'darkred',
                              'weight': 'normal',
                              'size': 14}

                if len(f_measured) != 0:
                    # if key is found print it and its offset colored red/green
                    tone, displaced = self.find(f_measured=f_measured[0])
                    if tone:
                        displayed_text = \
                            "{0:s} offset={1:.0f} cent".format(tone, displaced)
                        color = 'green' if displaced >= 0 else 'red'

                """Matplotlib block"""
                _start = timeit.default_timer()
                if _firstplot:
                    # Setup figure, axis, lines, text and initiate plot once
                    # and copy background
                    fig = plt.gcf()
                    ax1 = fig.add_subplot(111)
                    fig.set_size_inches(12, 6)
                    ln1, = ax1.plot(t1, yfft)
                    text = ax1.text(self.fmax, np.max(yfft), "",
                                    verticalalignment='top',
                                    horizontalalignment='right',
                                    fontsize=12,
                                    fontweight='bold'
                                    )
                    text1 = ax1.text(0., np.max(yfft), "",
                                     horizontalalignment='left',
                                     verticalalignment='top')
                    ax1.set_title(label=displayed_title,
                                  loc='right',
                                  fontdict=font_title)
                    ax1.set_xlabel('Frequency/Hz')
                    ax1.set_ylabel('Intensity/arb. units')
                    ax1background = fig.canvas.copy_from_bbox(ax1.bbox)
                else:
                    ln1.set_xdata(t1)
                    ln1.set_ydata(yfft)
                # set text attributes of lower subplot
                ax1.set_xlim([0., self.fmax])
                text.set_text(displayed_text)
                text.set_color(color)
                text.set_x(self.fmax)
                text.set_y(np.max(yfft))
                text1.set_text(info_text)
                text1.set_color(info_color)
                text1.set_x(0.)
                text1.set_y(np.max(yfft))

                # remove all collections: last object first (reverse)
                while ax1.collections:
                    ax1.collections.pop()
                yevents = EventCollection(positions=peaklist,
                                          color='tab:orange',
                                          linelength=0.05 * np.max(yfft),
                                          linewidth=2.
                                          )
                ax1.add_collection(yevents)
                yevents1 = EventCollection(positions=f_measured,
                                           color='tab:red',
                                           linelength=0.05 * np.max(yfft),
                                           lineoffset=-0.04 * np.max(yfft),
                                           linewidth=2.
                                           )
                ax1.add_collection(yevents1)

                # Rescale the axis so that the data can be seen in the plot
                # if you know the bounds of your data you could just set this
                # once, such that the axis don't keep changing
                ax1.relim()
                ax1.autoscale_view()

                if _firstplot:
                    plt.pause(0.0001)
                    _firstplot = False
                else:
                    # restore background
                    fig.canvas.restore_region(ax1background)
                    # redraw just the points
                    ax1.draw_artist(ln1)
                    ax1.draw_artist(text)
                    ax1.draw_artist(text1)
                    # fill in the axes rectangle
                    fig.canvas.blit(ax1.bbox)

                fig.canvas.flush_events()
                # resume the audio streaming, expect some retardation for the
                # status change

                _stop = timeit.default_timer()
                logging.debug("time utilized for matplotlib [s]: {0}".format(
                    str(_stop - _start)))

        return self.rc


def main():
    for tune in tuningtable.keys():
        print("Temperament ({1:d}) {0:s}".format(tune,
                                                 list(tuningtable).index(tune)))

    a = Tuner(
        tuning=list(tuningtable.keys())[int(input("Temperament [no]?: "))],
        a1=float(input("A4 pitch frequency [Hz]?: ")))

    h = keyboard.GlobalHotKeys({
        '<ctrl>+x': a.on_activate_x,
        '<ctrl>+y': a.on_activate_y,
        '<ctrl>+j': a.on_activate_j,
        '<ctrl>+k': a.on_activate_k,
        '<ctrl>+m': a.on_activate_m,
        '<ctrl>+n': a.on_activate_n,
        '<esc>': a.on_activate_esc})
    h.start()

    a.animate()
    plt.close('all')

    return 0
