#!/usr/bin/env python

"""
FFTonLiveAudio Copyright (C) 2020-21, Dr. Ralf Antonius Timmermann

A graphical tuning tool for string instruments, such as harpsichords and pianos.

Collects an audio signal from the input stream, that runs through a FFT. In
the frequency domain peak finding is applied. All peaks found are identified
as common partials (overtones) to one fundamental. It is compared to a tuning
table (feel free to enhance special non-equal tunings yourself) and a given
pitch value (input value). The deviation in units of cent is shown, too low
(red), too high (green).

Inharmonicity of strings is considered through equation
f_n = n * f_1 * sqrt(1 + B * n**2), where n = 1, 2, 3, ... and
B is the inharmonicity coefficient. The maximum inharmonicity accepted is
defined in parameters.py Change accordingly for harpsichords and pianos.

The hotkey 'ctrl-y' and 'x' stops the program and toggles between halt and
resume, respectively. 'Ctrl-j' and 'ctrl-k' shorten and lengthen the shift
between the audio slices, whereas 'ctrl-n' ('alt-n') and 'ctrl-m' ('alt-m')
diminish and increase the max (min) frequency displayed. 'ctrl-r' reset
parameter to initial values. Parameters, such as noise level, can be adjusted
in the parameters.py file.

This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it under
certain conditions.
"""

import pyaudio
from numpy import frombuffer, int16, hstack, log2
import matplotlib.pyplot as plt
from skimage import util
from timeit import default_timer
from time import sleep
from pynput import keyboard
from operator import itemgetter
from multiprocessing import Queue
import logging

from .tuningTable import tuningtable
from .FFTroutines import fft, peak, harmonics
from .multiProcess_matplot import MPmatplot
from Tuning import parameters

"""
2021/08/14 - Ralf A. Timmermann
- Update to version 2.1
    * new hotkeys to change minimum frequency in Fourier spectrum
2021/08/26 - Ralf A. Timmermann
- version 2.2
    * nested pie to display deviation of key from target value, parameter.PIE
    to toggle to previous setup
    * hotkey to reset to initial values
    * inner pie filled with color to indicate correct tuning
    * DEBUG: order peak list by frequency (ascending)
    * DEBUG: utilized time in ms
    * INFO: best results with f_1 instead of f_0
    * import solely modules to be used  
2022/02/26 - Ralf A. Timmermann
- version 2.3
    * f0 and b are derived from combinations of partitials with no common 
    divisor. In ambiguous cases the calculated partials' frequencies  
    (as derived from f0 and b) are compared with those measured and the cost 
    function (l1-norm) for a LASSO regression is minimized.
    * hotkey 'x' to toggle between halt and resume.
    * matplotlib commands are swapped to new superclass MPmatplot in 
    multiProcess_matplot.py that will be started in a proprietary process, 
    variables in dict passed through queue.
    * new plot parameters moved to parameters.py 
"""

__author__ = "Dr. Ralf Antonius Timmermann"
__copyright__ = "Copyright (C) Dr. Ralf Antonius Timmermann"
__credits__ = ""
__license__ = "GPLv3"
__version__ = "2.3.4"
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
        self.x = True
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
        audio_data = frombuffer(in_data, dtype=int16)
        self.callback_output.append(audio_data)

        return None, pyaudio.paContinue

    def on_activate_x(self):
        # toggle between suspending the audio stram and freezing the plat and
        # resuming: ON: True, OFF: False
        if self.x:
            print("resume with 'x' or quit with 'ctrl-y'")
        self.x = not self.x

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
        self.fmin -= parameters.FREQUENCY_STEP if self.fmin > 1500 else 100
        if self.fmin < 0:
            self.fmin = 0
        print("Min frequency displayed: {0:1.0f} Hz".format(self.fmin))

    def on_activate_ma(self):
        # increases the min. frequency plotted
        self.fmin += parameters.FREQUENCY_STEP if self.fmin >= 1500 else 100
        if self.fmin > 14500:
            self.fmin = 14500
        print("Min frequency displayed: {0:1.0f} Hz".format(self.fmin))
        if self.fmax - self.fmin < parameters.FREQUENCY_WIDTH_MIN:
            self.on_activate_m()

    def on_activate_n(self):
        # decreases the max. frequency plotted
        self.fmax -= parameters.FREQUENCY_STEP if self.fmax > 2000 else 100
        if self.fmax < 500:
            self.fmax = 500
        print("Max frequency displayed: {0:1.0f} Hz".format(self.fmax))
        if self.fmax - self.fmin < parameters.FREQUENCY_WIDTH_MIN:
            self.on_activate_na()

    def on_activate_m(self):
        # increases the max. frequency plotted
        self.fmax += parameters.FREQUENCY_STEP if self.fmax >= 2000 else 100
        if self.fmax > 15000:
            self.fmax = 15000
        print("Max frequency displayed: {0:1.0f} Hz".format(self.fmax))

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
            _stop = default_timer()
            logging.debug("time utilized for key finding: {0:.2f} ms".format(
                (_stop - _start) * 1000.))

        _start = default_timer()

        for i in range(-4, 5):  # key range from keys C0 till B8
            offset = log2(f_measured / self.a1) * 1200 - i * 1200
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
        # interrupt/resume on hotkey 'x' and clear buffer after resuming.
        if not self.x:
            self.stream.stop_stream()
            while not self.x:
                # loop and wait until 'x' or 'ctrl-y' is pressed
                if self.rc == 'y':
                    return None
                sleep(.1)
            self.callback_output = list()
            self.stream.start_stream()
            logging.info(
                "Clearing audio buffer and resuming audio stream ...")
        logging.debug("=== new audio cycle: filling buffer ===")
        _start = default_timer()
        # wait until buffer filled by at least one FFT slice, where
        # length is in units of buffer = 1024
        while len(self.callback_output) < parameters.SLICE_LENGTH // 1024:
            sleep(0.02)
        # Convert the list of numpy-arrays into a 1D array (column-wise)
        amp = hstack(self.callback_output)
        slices = util.view_as_windows(amp,
                                      window_shape=(
                                          parameters.SLICE_LENGTH,),
                                      step=self.step)
        _stop = default_timer()
        logging.debug("Audio shape: {0}, Sliced audio shape: {1}"
                      .format(amp.shape,
                              slices.shape))
        logging.debug("time utilized for Audio: {0:.2f} ms".format(
            (_stop - _start) * 1000.))

        return slices

    def animate(self):
        """
        calling routine for audio, FFT, peak and partials and key finding,
        and plotting. Listens for events in plot window
        :return:
        string
            return code
        """
        queue = Queue()
        _process = MPmatplot(queue=queue,
                             a1=self.a1,
                             tuning=self.tuning
                             )
        _process.start()
        f_measured = list()

        # start Recording
        self.stream.start_stream()
        logging.info(
            "Permit a few cycles to adjust the audio device for sound input")

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
                # call peakfinding
                peaks = peak(frequency=t1,
                             spectrum=yfft)
                peaklist = list(map(itemgetter(0), peaks))
                # call harmonics
                if peaks is not None:
                    f_measured = harmonics(peaks=peaks)  # find the key
                # call key finding
                if len(f_measured) != 0:
                    # if key is found print it and its offset colored red/green
                    key, off = self.find(f_measured=f_measured[0])
                else:
                    off = 0.
                    key = ''
                # send params into queue for plotting
                queue.put(
                    {'yfft': yfft,
                     'key': key,
                     'off': off,
                     'slices': slices,
                     'step': self.step,
                     'fmin': self.fmin,
                     'fmax': self.fmax,
                     'peaklist': peaklist,
                     'f_measured': f_measured}
                )

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
        'x': a.on_activate_x,         # toggle between halt & resume
        'q': a.on_activate_y,         # exit through closing plot window
        '<ctrl>+y': a.on_activate_y,  # exit
        '<ctrl>+w': a.on_activate_y,  # exit
        '<alt>+w': a.on_activate_y,   # exit
        '<ctrl>+j': a.on_activate_j,  # decrease slices shift
        '<ctrl>+k': a.on_activate_k,  # increase slices shift
        '<ctrl>+r': a.on_activate_r,  # reset parameter to initial values
        '<ctrl>+m': a.on_activate_m,  # increase max freq
        '<ctrl>+n': a.on_activate_n,  # decrease max freq
        '<alt>+m': a.on_activate_ma,  # increase min freq
        '<alt>+n': a.on_activate_na   # decrease min freq
        })
    h.start()

    a.animate()
    plt.close('all')

    return 0
