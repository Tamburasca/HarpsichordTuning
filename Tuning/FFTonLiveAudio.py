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

The hotkey ctrl-y and ctrl-x exits and stops the program, respectively,
ESC to resume. Ctrl-j and ctrl-k shorten and lengthen the shift between the
audio slices, whereas ctrl-n (alt-n) and ctrl-m (alt-m) diminish and
increase the max (min) frequency displayed. Parameters can be adjusted in the
parameters.py file.

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

"""
2021/ß8/14 - Ralf A. Timmermann <rtimmermann@gmx.de>
- Update to version 2.1
    * new hotkeys to change minimum frequency in Fourier spectrum
2021/ß8/18 - Ralf A. Timmermann <rtimmermann@gmx.de>
- Update to version 2.2
    * nested pie to display deviation of key from target value, parameter.PIE
    to toggle to previous setup
    * hotkey to reset to initial values
    * inner pie filled with color to indicate correct tuning
    * DEBUG: order peak list by frequency (ascending)  
"""

__author__ = "Dr. Ralf Antonius Timmermann"
__copyright__ = "Copyright (C) Dr. Ralf Antonius Timmermann"
__credits__ = ""
__license__ = "GPLv3"
__version__ = "2.2.4"
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
        self.step: int = parameters.SLICE_SHIFT
        self.fmax: int = parameters.FREQUENCY_MAX
        self.fmin: int = 0
        self.a1 = kwargs.get('a1')
        self.tuning = kwargs.get('tuning')  # see tuningTable.py
        self.v = True
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

    def on_activate_v(self):
        # toggle, not used to date
        self.v = not self.v

    def on_activate_x(self):
        # suspend the audio stram and freezes the plat
        print("continue with ESC or quit with ctrl-y")
        self.rc = 'x'

    def on_activate_y(self):
        # exits the program
        print("quitting...")
        self.rc = 'y'

    def on_activate_r(self):
        # reset parameters to initial values
        print("reseting parameters...")
        self.fmin = 0
        self.fmax = parameters.FREQUENCY_MAX
        self.step = parameters.SLICE_SHIFT

    def on_activate_k(self):
        # increases the shift by which the slices progress
        self.step += 1024
        if self.step > parameters.SLICE_LENGTH:
            self.step = parameters.SLICE_LENGTH
        print("Slice shift: {0:d} samples".format(self.step))

    def on_activate_j(self):
        # decreases the shift by which the slices progress
        self.step -= 1024
        if self.step < 4096:
            self.step = 4096
        print("Slice shift: {0:d} samples".format(self.step))

    def on_activate_na(self):
        # decreases the min. frequency plotted
        self.fmin -= 500 if self.fmin > 1500 else 100
        if self.fmin < 0:
            self.fmin = 0
        print("Min frequency displayed: {0:1.0f}Hz".format(self.fmin))

    def on_activate_ma(self):
        # increases the min. frequency plotted
        self.fmin += 500 if self.fmin >= 1500 else 100
        if self.fmin > 14500:
            self.fmin = 14500
        print("Min frequency displayed: {0:1.0f}Hz".format(self.fmin))
        if self.fmax - self.fmin < 500:
            self.on_activate_m()

    def on_activate_n(self):
        # decreases the max. frequency plotted
        self.fmax -= 500 if self.fmax > 2000 else 100
        if self.fmax < 500:
            self.fmax = 500
        print("Max frequency displayed: {0:1.0f}Hz".format(self.fmax))
        if self.fmax - self.fmin < 500:
            self.on_activate_na()

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
        return '', 0

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

        logging.debug("=== new cycle ===")
        _start = timeit.default_timer()
        # wait until buffer filled by at least one FFT slice, where
        # length is in units of buffer = 1024
        while len(self.callback_output) < parameters.SLICE_LENGTH // 1024:
            time.sleep(0.02)
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
        f_measured = list()
        fig = None
        ln1 = None
        ax1 = None
        text, text1 = None, None
        inset_pie = None
        ax1background = None

        # matplotlib subroutine for pie inlet
        def pie(axes, displaced, key_pressed):
            # delete all patches and texts from inset_pie axes that piled up
            # print(axes.__dict__)
            while axes.patches:
                axes.patches.pop()
            while axes.texts:
                axes.texts.pop()
            axes.text(x=0,
                      y=0,
                      s=key_pressed,
                      fontdict={'fontsize': 20,
                                'horizontalalignment': 'center',
                                'verticalalignment': 'center'})
            # outer pie
            axes.pie(
                [-displaced, 100 + displaced] if displaced < 0 else [
                    displaced, 100 - displaced],
                startangle=90,
                colors=['red' if displaced < 0 else 'green', 'white'],
                counterclock=displaced < 0,
                labels=(
                    "{0:.0f} cent".format(displaced) if key_pressed else '', ''
                ),
                wedgeprops=dict(width=.6,
                                edgecolor='k',
                                lw=.5))
            # inner pie
            axes.pie([1],
                     # 2 cents within the target means key is well tuned
                     # paint pie white (default) to make it opaque
                     colors='y' if key_pressed and -2 < displaced < 2 else 'w',
                     radius=.4
                     )

            return

        # vertical bar subroutine
        def eventcollection(axes, peak_list, f_meas):
            # remove all previous collections from axes, reverse order
            while axes.collections:
                axes.collections.pop()
            y_axis0, y_axis1 = axes.get_ylim()
            if parameters.DEBUG:
                yevents = EventCollection(
                    positions=peak_list,
                    color='tab:orange',
                    lineoffset=(y_axis0 + y_axis1) / 2,
                    linelength=np.abs(y_axis0) + y_axis1,
                    linewidth=1.
                )
            else:
                yevents = EventCollection(
                    positions=peak_list,
                    color='tab:orange',
                    linelength=-2 * y_axis0,
                    lineoffset=0.,
                    linewidth=2.
                )
            axes.add_collection(yevents)
            yevents1 = EventCollection(positions=f_meas,
                                       color='tab:red',
                                       linelength=-2 * y_axis0,
                                       lineoffset=y_axis0,
                                       linewidth=2.
                                       )
            axes.add_collection(yevents1)

            return

        _firstplot = True
        plt.ion()  # Stop matplotlib windows from blocking

        resolution = parameters.RATE / parameters.SLICE_LENGTH * 1.5
        logging.debug(
            "Resolution incl. Hanning apodization (Hz/channel) ~ {0}"
            .format(str(resolution)))

        # start Recording
        self.stream.start_stream()
        logging.info(
            "Permit a few cycles to configure the audio device for sound input")
        # main loop while audio stream active
        while self.stream.is_active():
            slices = self.slice()
            if self.rc == 'y':
                return self.rc

            # work off all slices, before pulling from audio stream
            for sl in slices:
                logging.debug("no of slices: " + str(len(slices)))
                # remove current slice from beginning of buffer
                del self.callback_output[0:self.step // 1024]
                # calculate FFT
                t1, yfft = fft(amp=sl)
                ymax = np.max(yfft)
                # call peakfinding
                peaks = peak(frequency=t1,
                             spectrum=yfft)
                peaklist = list(map(itemgetter(0), peaks))
                # call harmonics
                if peaks is not None:
                    f_measured = harmonics(peaks=peaks)  # find the key

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
                off_text = ''
                off_color = None
                if len(f_measured) != 0:
                    # if key is found print it and its offset colored red/green
                    key, off = self.find(f_measured=f_measured[0])
                    if key and not parameters.PIE:
                        off_text = \
                            "{0:s} offset={1:.0f} cent".format(key, off)
                        off_color = 'green' if off >= 0 else 'red'
                else:
                    off = 0.
                    key = ''

                # Matplotlib block
                _start = timeit.default_timer()
                if _firstplot:
                    # Setup figure, axis, lines, text and initiate plot once
                    # and copy background
                    fig = plt.gcf()
                    ax1 = fig.add_subplot(111)
                    fig.set_size_inches(12, 6)
                    fig.canvas.set_window_title(
                        'Digital String Tuner (c) Ralf A. Timmermann')
                    if parameters.PIE:
                        # inset_axes with nested pie and equal aspect ratio
                        inset_pie = ax1.inset_axes(
                            bounds=[0.65, 0.5, 0.35, 0.5],
                            zorder=5)  # default
                        inset_pie.axis('equal')
                    # define plot
                    ln1, = ax1.plot(t1, yfft)
                    text = ax1.text(self.fmax, ymax, '',
                                    verticalalignment='top',
                                    horizontalalignment='right',
                                    fontsize=12,
                                    fontweight='bold'
                                    )
                    text1 = ax1.text(self.fmin, ymax, '',
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

                # set attributes of subplot
                ax1.set_xlim([self.fmin, self.fmax])
                # permit some percentages of margin to the x-axes
                ax1.set_ylim([-0.04 * ymax, 1.025 * ymax])
                text.set_x(self.fmax)
                text.set_y(ymax)
                if parameters.PIE:
                    # call nested pie inset
                    pie(axes=inset_pie,
                        displaced=off,
                        key_pressed=key)
                else:
                    # plain text
                    text.set_text(off_text)
                    text.set_color(off_color)
                text1.set_text(info_text)
                text1.set_color(info_color)
                text1.set_x(self.fmin)
                text1.set_y(ymax)
                # plot vertical bars
                eventcollection(ax1, peaklist, f_measured)

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

                # resume audio streaming, expect retardation for status change
                fig.canvas.flush_events()

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
        a1=float(input("A4 pitch frequency [Hz]?: "))
    )

    h = keyboard.GlobalHotKeys({
        '<ctrl>+v': a.on_activate_v,  # toggle not used to date
        '<ctrl>+x': a.on_activate_x,  # halt
        '<ctrl>+y': a.on_activate_y,  # exit
        '<ctrl>+j': a.on_activate_j,  # decrease slices shift
        '<ctrl>+k': a.on_activate_k,  # increase slices shift
        '<ctrl>+r': a.on_activate_r,  # reset parameter to initial values
        '<ctrl>+m': a.on_activate_m,  # increase max freq
        '<ctrl>+n': a.on_activate_n,  # decrease max freq
        '<alt>+m': a.on_activate_ma,  # increase min freq
        '<alt>+n': a.on_activate_na,  # decrease min freq
        '<esc>': a.on_activate_esc,  # resume
        'q': a.on_activate_y})  # exit through closing plot window
    h.start()

    a.animate()
    plt.close('all')

    return 0
