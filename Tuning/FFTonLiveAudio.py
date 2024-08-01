#!/usr/bin/env python3

"""
FFTonLiveAudio Copyright (C) 2020-24, Dr. Ralf Antonius Timmermann

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
from numpy import frombuffer, int16, hstack, log2, zeros_like, sqrt
from scipy.signal import butter, sosfilt
from skimage import util
from time import sleep
from pynput import keyboard
from operator import itemgetter
from multiprocessing import Queue
import logging
from typing import Tuple, Dict
from numpy.typing import NDArray
import time
# internal
from tuningTable import tuningtable
from FFTroutines import fft
from FFTpeaks import peak
from FFTharmonics import harmonics
from multiProcess_matplot import MPmatplot
from FFTaux import mytimer, baseline_als_optimized
import parameters

__author__ = "Dr. Ralf Antonius Timmermann"
__copyright__ = "Copyright (c) 2020-24 Dr. Ralf Antonius Timmermann"
__credits__ = ""
__license__ = "GPLv3"
__version__ = "3.5.0"
__maintainer__ = "Dr. Ralf A. Timmermann"
__email__ = "ralf.timmermann@gmx.de"
__status__ = "Prod"

print(__doc__)

logging.basicConfig(format=parameters.myformat,
                    level=logging.INFO,
                    datefmt="%H:%M:%S")
if parameters.DEBUG:
    logging.getLogger().setLevel(logging.DEBUG)


class Tuner:
    def __init__(
            self,
            a1: float,
            tuning: str
    ):
        """
        :param a1: float
            pitch frequency for a1
        :param tuning: string
            tuning temperament
        """
        CHUNKSIZE: int = 1024  # fixed chunk size
        self.step: int = parameters.SLICE_SHIFT
        self.fmax: int = parameters.FREQUENCY_MAX
        self.fmin: int = 0
        self.noise_level = parameters.NOISE_LEVEL
        self.a1: float = a1
        self.tuning: float = tuning  # see tuningTable.py
        self.x: bool = True
        self.rc: str = None
        self.noise_toggle = False
        self.__n: int = 0
        self.baseline = None
        self.std = None
        self.callback_output: List = []

        audio = pyaudio.PyAudio()
        self.stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=parameters.RATE,
            output=False,
            input=True,
            stream_callback=self.callback,
            frames_per_buffer=CHUNKSIZE
        )
        logging.debug("Audio Device info: {}".
                      format(audio.get_default_input_device_info()))

    def callback(
            self,
            in_data: bytes,
            frame_count: int,
            time_info: Dict,
            flag: int
    ) -> Tuple[None, int]:
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

    def on_activate_x(self) -> None:
        # toggle between suspending the audio stram and freezing the plat and
        # resuming: ON: True, OFF: False
        if self.x:
            print("Resume with 'ctrl-x' or 'ctrl-y' to quit.")
        self.x = not self.x

    def on_activate_y(self) -> None:
        # exits the program
        print("quitting...")
        self.rc = 'y'

    def on_activate_r(self) -> None:
        # reset parameters to initial values
        print("reseting parameters...")
        self.fmin = 0
        self.fmax = parameters.FREQUENCY_MAX
        self.step = parameters.SLICE_SHIFT
        self.noise_level = parameters.NOISE_LEVEL
        self.noise_toggle = False
        self.__n = 0
        self.baseline = None
        self.std = None

    def on_activate_k(self) -> None:
        # increases the shift by which the slices progress
        self.step += 1024
        if self.step > parameters.SLICE_LENGTH:
            self.step = parameters.SLICE_LENGTH
        print("Slice shift: {0:d} samples".format(self.step))

    def on_activate_j(self) -> None:
        # decreases the shift by which the slices progress
        self.step -= 1024
        if self.step < 4096:
            self.step = 4096
        print("Slice shift: {0:d} samples".format(self.step))

    def on_activate_na(self) -> None:
        # decreases the min. frequency plotted
        self.fmin -= parameters.FREQUENCY_STEP if self.fmin > 1500 else 100
        if self.fmin < 0:
            self.fmin = 0
        print("Min frequency displayed: {0:1.0f} Hz".format(self.fmin))

    def on_activate_ma(self) -> None:
        # increases the min. frequency plotted
        self.fmin += parameters.FREQUENCY_STEP if self.fmin >= 1500 else 100
        if self.fmin > 14500:
            self.fmin = 14500
        print("Min frequency displayed: {0:1.0f} Hz".format(self.fmin))
        if self.fmax - self.fmin < parameters.FREQUENCY_WIDTH_MIN:
            self.on_activate_m()

    def on_activate_n(self) -> None:
        # decreases the max. frequency plotted
        self.fmax -= parameters.FREQUENCY_STEP if self.fmax > 2000 else 100
        if self.fmax < 500:
            self.fmax = 500
        print("Max frequency displayed: {0:1.0f} Hz".format(self.fmax))
        if self.fmax - self.fmin < parameters.FREQUENCY_WIDTH_MIN:
            self.on_activate_na()

    def on_activate_m(self) -> None:
        # increases the max. frequency plotted
        self.fmax += parameters.FREQUENCY_STEP if self.fmax >= 2000 else 100
        if self.fmax > 15000:
            self.fmax = 15000
        print("Max frequency displayed: {0:1.0f} Hz".format(self.fmax))

    def on_activate_noise_up(self) -> None:
        # increase noise level by 10%
        self.noise_level *= 1.1
        print("Noise level increased to {0:1.1f}".format(self.noise_level))

    def on_activate_noise_down(self) -> None:
        # decrease noise level by 9.09%
        self.noise_level /= 1.1
        print("Noise level decreased to {0:1.1f}".format(self.noise_level))

    def on_activate_measure_noise(self) -> None:
        self.noise_toggle = not self.noise_toggle
        if self.noise_toggle: print("Measuring Noise Level. Please keep quiet!")

    @mytimer
    def noise_threshold(
            self,
            yfft: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        yfft averaged and standard deviation per bin by adding new dataset on previous:
        Welford’s method is a usable single-pass method for computing the variance.
        https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
        https://math.stackexchange.com/questions/775391/can-i-calculate-the-new-standard-deviation-when-adding-a-value-without-knowing-t
        :param yfft: NDArray
        :return: NDArray, NDArray
        """
        self.__n += 1
        if self.__n == 1:
            self.__av = yfft
            return None, None
        elif self.__n == 2:
            yfft_first = self.__av
            self.__av = (yfft_first + yfft) / 2.
            self.__std_squared = (
                    (yfft - self.__av) ** 2 + (yfft_first - self.__av) ** 2
            )
        else:
            av_previous = self.__av
            self.__av = ((self.__n - 1) * self.__av + yfft) / self.__n
            self.__std_squared = (
                    ((self.__n - 2) * self.__std_squared +
                     (yfft - self.__av) * (yfft - av_previous)) / (self.__n - 1)
            )
        print("Noise Measurement ..., Iteration No: {}".format(self.__n - 1))

        return self.__av, sqrt(self.__std_squared)

    @mytimer("key finding & absolute pitch level")
    def find(self, f_measured: float) -> Tuple[str, float]:
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
        for i in range(-4, 5):  # key range from keys C0 till B8
            offset = log2(f_measured / self.a1) * 1200 - i * 1200
            for key, value in tuningtable[self.tuning].items():
                displaced = offset + tuningtable[self.tuning].get('A') - value
                if -60 < displaced < 60:
                    logging.debug("{0} {1} {2} {3}".format(
                        str(i),
                        str(key),
                        str(value),
                        str(displaced
                            + tuningtable[self.tuning].get('A')
                            - value))
                    )
                    # matplotlib formate
                    key_pressed = "{0}$_{{{1}}}$".format(key, str(i + 4))

                    return key_pressed, displaced

        return '', 0.

    @mytimer("audio")
    def slice(self) -> NDArray:
        """
        collects the audio data and arranges it into slices of length
        SLICES_LENGTH shifted by SLICES_SHIFT
        :return:
        NDArray
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
        # wait until buffer filled by at least one FFT slice, where
        # length is in units of buffer = 1024
        while len(self.callback_output) < parameters.SLICE_LENGTH // 1024:
            sleep(0.02)
        # Convert the list of numpy-arrays into a 1D array (column-wise)
        amp = hstack(self.callback_output)
        slices = util.view_as_windows(amp,
                                      window_shape=(
                                          parameters.SLICE_LENGTH),
                                      step=self.step)
        logging.debug("Audio shape: {0}, Sliced audio shape: {1}"
                      .format(amp.shape,
                              slices.shape))

        return slices

    def animate(self) -> str:
        """
        calling routine for audio, FFT, peak and partials and key finding,
        and plotting. Listens for events in plot window
        :return:
        string
            return code
        """
        sos = butter(
            N=parameters.F_ORDER,
            Wn=parameters.F_FILT,
            btype='highpass',
            fs=parameters.RATE,
            output='sos'
        )
        def highpass_filter(sig: NDArray) -> NDArray:
            return sosfilt(sos, sig)

        queue = Queue()
        _process = MPmatplot(queue=queue,
                             a1=self.a1,
                             tuning=self.tuning
                             )
        _process.start()
        f_measured: List = []

        # start Recording
        self.stream.start_stream()
        logging.info(
            "Permit a few cycles to adjust audio device!")

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
                # apply highpass filter on time series
                #sl = highpass_filter(sl)
                # calculate FFT
                t1, yfft = fft(amp=sl)
                # measure noise if toggled
                if self.noise_toggle:
                    self.baseline, self.std = self.noise_threshold(yfft)
                # other option
                # baseline = baseline_als_optimized(yfft, lam=3.e4, p=.01, niter=1)
                # call peakfinding
                peaks = peak(frequency=t1,
                             spectrum=yfft,
                             baseline=self.baseline,
                             std=self.std,
                             noise_level=self.noise_level)
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
                     'noise_toggle': self.noise_toggle,
                     'baseline':
                         self.baseline
                         + parameters.FACTOR_STANDARD_DEV_NOISE * self.std
                            if self.baseline is not None else None,
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


def main() -> int:

    def input_pitch() -> float:
        pitch = float(input("A4 pitch frequency [Hz]?: "))
        assert 392 < pitch < 494
        return pitch

    for tune in tuningtable.keys():
        print("Temperament ({1:d}) {0:s}".format(
            tune,
            list(tuningtable).index(tune))
        )
    a = None
    while a is None:
        try:
            a = Tuner(
                tuning=list(tuningtable.keys())[int(
                    input("Temperament [no.]?: "))],
                a1=input_pitch()
            )
        except (IndexError, ValueError):
            print("Error: Chosen number not in list of provided temperaments")
        except AssertionError:
            print("Error: A4 pitch in the range from 392 to 494 Hz")
        except KeyboardInterrupt:
            return 1

    h = keyboard.GlobalHotKeys({
        '<ctrl>+y': a.on_activate_y,  # exit
        # '<ctrl>+w': a.on_activate_y,  # exit
        '<ctrl>+x': a.on_activate_x,  # toggle halt/resume
        '<ctrl>+j': a.on_activate_j,  # decrease slice shift
        '<ctrl>+k': a.on_activate_k,  # increase slice shift
        '<ctrl>+r': a.on_activate_r,  # reset parameter to initial values
        '<ctrl>+m': a.on_activate_m,  # increase max freq
        '<ctrl>+n': a.on_activate_n,  # decrease max freq
        '<alt>+m': a.on_activate_ma,  # increase min freq
        '<alt>+n': a.on_activate_na,  # decrease min freq
        '<ctrl>+<alt>+1': a.on_activate_noise_down,  # decrease noise level
        '<ctrl>+<alt>+2': a.on_activate_noise_up,  # increase noise level
        '<ctrl>+<alt>+3': a.on_activate_measure_noise # toggle noise measurement on/off
    })
    h.start()
    a.animate()

    return 0
