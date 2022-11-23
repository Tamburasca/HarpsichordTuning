#!/usr/bin/env python

"""
FFTonLiveAudio Copyright (C) 2020-22, Dr. Ralf Antonius Timmermann

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
from numpy import frombuffer, int16, hstack, log2, ndarray, zeros_like, sqrt
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt
from skimage import util
from time import sleep
from pynput import keyboard
from operator import itemgetter
from multiprocessing import Queue
import logging
from typing import Tuple
# internal
from Tuning.tuningTable import tuningtable
from Tuning.FFTroutines import fft
from Tuning.FFTpeaks import peak
from Tuning.FFTharmonics import harmonics
from Tuning.multiProcess_matplot import MPmatplot
from Tuning.FFTaux import mytimer, baseline_als_optimized
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
    * consumed time per module in DEBUG mode measured in wrapper mytimer
    * FFTaux.py added
2022/04/10 - Ralf A. Timmermann
- version 3.0 (productive)
    * Finding the NMAX highes peaks in the frequency spectrum: prominence 
    for find_peaks disabled, peak heights minus background calculated by 
    utilizing the average interpolated positions of left and right intersection 
    points of a horizontal line at the respective evaluation height.
    * modified global hot keys, such as q is disabled.
2022/07/17 - Ralf A. Timmermann
- version 3.1.0 (productive)
    * slice shift is fixed value in parameter file
    * import absolute paths
    * final L1 minimization on difference between measured and calculated 
    partials utilizing scipy.optimize.minimize through brute
    to determine base-frequency and inharmonicity (toggled in parameters)
2022/07/19 - Ralf A. Timmermann
- version 3.1.1 (productive)
    * Noise level can be adjusted through global hot keys
    * L1 minimization called only if more than 2 measured peaks
    * catching if erroneous input from the keyboard
2022/07/20 - Ralf A. Timmermann
- version 3.1.2 (productive)
    * error correction: Jacobian sign of derivative toggled
    * L1_contours.py added to demonstrate L1 cost function and its Jacobian (
    not needed with brute force) 
2022/07/26 - Ralf A. Timmermann
- version 3.1.3 (productive)
    * slice of inharmonicity factor b is on log10 scale for equidistant grids
    for brute force minimizer
2022/08/07 - Ralf A. Timmermann
- version 3.2 (productive)
    * updated for Python v3.9
    * typing
2022/08/07 - Ralf A. Timmermann
- version 3.2.1 (productive)
    * SLSQP minimizer
2022/08/12 - Ralf A. Timmermann
- version 3.2.2 
    * absoute pitch in pie
2022/08/17 - Ralf A. Timmermann
- version 3.3.1 
    * dublicates removed from the partials list
    * for the minimizer: list of found frequencies is now tagged with 
    the appropriate partials, that makes l1 computation unambiguous and avoids
    local minima
    * code cleansing
2022/09/27 - Ralf A. Timmermann
- version 3.4.1
    * highpass filter parameters outsourced as global parameters
    * Noise measurement with no audio signal (silence) comprising mean and 
    standard deviation per bin - need to be toggled on/off per global hotkey
    (in experimental stage) 
"""

__author__ = "Dr. Ralf Antonius Timmermann"
__copyright__ = "Copyright (C) Dr. Ralf Antonius Timmermann"
__credits__ = ""
__license__ = "GPLv3"
__version__ = "3.4.1"
__maintainer__ = "Dr. Ralf A. Timmermann"
__email__ = "rtimmermann@astro.uni-bonn.de"
__status__ = "Prod"

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
        self.noise_level = parameters.NOISE_LEVEL
        self.a1: float = kwargs.get('a1')
        self.tuning: float = kwargs.get('tuning')  # see tuningTable.py
        self.x: bool = True
        self.rc: str = None
        self.noise_toggle = False
        self.__n: int = 0
        self.baseline = None
        self.std = None

        self.callback_output = list()
        audio = pyaudio.PyAudio()
        self.stream = audio.open(format=pyaudio.paInt16,
                                 channels=1,
                                 rate=parameters.RATE,
                                 output=False,
                                 input=True,
                                 stream_callback=self.callback)

    def callback(self, in_data, frame_count, time_info, flag) -> Tuple[None, int]:
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
            print("Resume with 'ctrl-x' or with 'ctrl-y' to quit.")
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
    def noise_threshold(self, yfft: ndarray) -> Tuple[ndarray, ndarray]:
        """
        yfft averaged and standard deviation per bin by adding new dataset on previous:
        Welford’s method is a usable single-pass method for computing the variance.
        https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
        https://math.stackexchange.com/questions/775391/can-i-calculate-the-new-standard-deviation-when-adding-a-value-without-knowing-t
        :param yfft: ndarray
        :return: ndarray, ndarray
        """
        self.__n += 1
        if self.__n == 1:
            self.__av = yfft
            return None, None
        elif self.__n == 2:
            yfft_first = self.__av
            self.__av = (yfft_first + yfft) / 2
            self.__std_squared = (yfft - self.__av) ** 2 + (yfft_first - self.__av) ** 2
        else:
            av_previous = self.__av
            self.__av = ((self.__n - 1) * self.__av + yfft) / self.__n
            self.__std_squared = ((self.__n - 2) * self.__std_squared +
                                  (yfft - self.__av) * (yfft - av_previous)) / (self.__n - 1)
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
                    logging.debug("{0} {1} {2} {3}"
                                  .format(str(i),
                                          str(key),
                                          str(value),
                                          str(displaced + tuningtable[self.tuning].get('A') - value)))
                    key_pressed = "{0}$_{{{1}}}$".format(key, str(i + 4)) # matplotlib formate

                    return key_pressed, displaced

        return '', 0.

    @mytimer("audio")
    def slice(self) -> ndarray:
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
        @mytimer
        def highpass_filter(sig: ndarray) -> ndarray: return sosfilt(sos, sig)

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
                     'baseline': 20. * self.std if self.baseline is not None else None,
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
        print("Temperament ({1:d}) {0:s}".format(tune,
                                                 list(tuningtable).index(tune)))
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
            pass
        except AssertionError:
            print("Error: A4 pitch in the range from 392 to 494 Hz")
            pass
        except KeyboardInterrupt:
            return 1

    h = keyboard.GlobalHotKeys({
        '<ctrl>+y': a.on_activate_y,  # exit
        '<ctrl>+w': a.on_activate_y,  # exit
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
        '<ctrl>+<alt>+3': a.on_activate_measure_noise
    })
    h.start()
    a.animate()

    plt.close('all')
    return 0
