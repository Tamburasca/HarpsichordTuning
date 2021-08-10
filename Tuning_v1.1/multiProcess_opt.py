from multiprocessing import Process, Queue
from scipy.optimize import curve_fit
import numpy as np
from operator import itemgetter
import logging
from Tuning import parameters


logging.basicConfig(format=parameters.myformat,
                    level=logging.INFO,
                    datefmt="%H:%M:%S")
if parameters.DEBUG:
    logging.getLogger().setLevel(logging.DEBUG)


class ThreadedOpt:
    """
    driver for multiprocessing:
    called by peak finding to locate exacter peak positions through minimizing
    a Gauss fit to each individual peak found. Each minimization call is
    performed by a discrete process (multiprocessing).
    """

    # initializations
    def __init__(self, freq, amp, initial):
        """
        :param freq: list of floats
            frequencies of the FFT
        :param amp:  list of floats
            amplitudes of the FFT
        :param initial: float tuple
            frequencies and corresponding heights of peaks found
        """
        self._amp = amp
        self._freq = freq
        self.num_threads = len(initial)
        self._x = list(map(itemgetter(0), initial))
        self._y = list(map(itemgetter(1), initial))

    # Run the optimization. Make the threads here.
    def run(self):

        queue = Queue()
        processes = []
        peaks = []
        logging.debug("Number of processes: " + str(self.num_threads))

        for thread_id in range(self.num_threads):  # Make the threads and start them off
            p = Process(target=self.fitting,
                        args=(queue, thread_id,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=.1)  # resume after timeout, cleanup later

        while not queue.empty():
            rc = queue.get()
            peaks.append([rc[0], rc[1]])
        queue.close()
        queue.join_thread()

        # clean up processes that did not join within timeout period
        while processes:
            p = processes.pop()
            if p.is_alive():
                p.terminate()
                logging.warning("Multiprocessing: Process did not join, cleaning up ...")

        # unsorted peaks
        return peaks

    # Each thread goes through this.
    def fitting(self, queue, thread_id):
        """
        :param queue: object
            common queue
        :param thread_id: int
            thread id = 0, 1, 2, ..., len(initial)-1
        :return: none
        """
        window = 6  # width on either side
        x = self._x[thread_id]
        y = self._y[thread_id]
        # do not exceed the array on either side
        if x-window < 0 or x+window > len(self._freq)-1:
            logging.warning('Fit initial value out of range: peak disregarded!')

            return

        # Gaussian width to be guessed better
        guess = [self._freq[x], y, 0.25]
        boundaries = ([self._freq[x] - window, 0.25*y, 0.],
                      [self._freq[x] + window, 2*y, 3.])
        try:
            popt_ind, pcov = curve_fit(self.gauss,
                                       self._freq[x-window:x+window],
                                       self._amp[x-window:x+window],
                                       p0=guess,
                                       bounds=boundaries,
                                       method='dogbox')
            logging.debug('Position (Hz): {0:e}, Height (arb. Units): {1:e}, FWHM (Hz): {2:e}'
                          .format(popt_ind[0], popt_ind[1], 2.354 * popt_ind[2]))
        except RuntimeError:
            logging.warning('Fit failure: peak disregarded!')

            return
        # put the results into the queue
        queue.put(popt_ind)

        return

    @staticmethod
    def gauss(x, *params):
        """
        superposition of multi-Gaussian curves
        :param x: list of floats
            x-values
        :param params:
            param[0]: x-value
            param[1]: amplitude
            param[2]: width
            param[0+i]: x-value
            ...
            where i = 0, 1, 2, ...
        :return:
            y-values of multiple Gaussing fit
        """
        # Gaussian fit with FFT
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            ctr = params[i]
            amp = params[i + 1]
            wid = params[i + 2]
            y = y + amp * np.exp(-0.5 * ((x - ctr) / wid) ** 2)

        return y
