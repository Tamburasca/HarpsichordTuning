from multiprocessing import Process, Queue
from scipy.optimize import minimize
import numpy as np
from operator import itemgetter
from .parameters import _debug, INHARM


class threaded_opt:
    """
    driver for multiprocessing:
    called from harmonics to maximize cross correlation. In this case the two
    1-dimensional sequences have been reduced to those elements not zero to
    speed up the code.
    Each minimization call is preformed by a discrete process (multiprocessing)
    We are still playing around with the optimization method, that sucks somehow
    """

    # initializations
    def __init__(self, amp, freq, initial):

        self.amp = amp
        self.freq = freq
        self.a = list(map(itemgetter(0), initial))
        self.b = list(map(itemgetter(1), initial))
        self.num_threads = len(initial)
        self.best_fun = None
        self.best_x = None

    # Run the optimization. Make the threads here.
    @property
    def run(self):

        self.queue = Queue()
        processes = []

        for x in range(self.num_threads):  # Make the threads and start them off
            p = Process(target=self.target_function,
                        args=(x,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=1)

        self.best_fun = 1.e36
        for _ in processes:
            chi, x = self.queue.get()
            if chi < self.best_fun:
                self.best_x = x
                self.best_fun = chi
        self.queue.close()
        self.queue.join_thread()

        # clean up processes that did not join within timeout period
        while processes:
            p = processes.pop()
            if p.is_alive():
                p.terminate()
                print("Process did not join!  Cleaning up ...")

        return None

    # Each thread goes through this.
    def target_function(self, thread_ID):

        # set the boundaries for f_0 within 2% and b < INHARM
        def bnds(x):
            return ((0.98 * x, 1.02 * x), (0, INHARM))

        result = minimize(self.chi_square,
                          [self.a[thread_ID], self.b[thread_ID]],
                          bounds=bnds(self.a[thread_ID]),
                          method="Powell",  # "nelder-mead"
                          options={'xtol': 1.e-7}  # still need to fiddle with xtol
                         )
        self.queue.put((result.fun, result.x))

    # maximize the cross-correlation (negate result for minimizer)
    def chi_square(self, x):
        r = 0
        _i = 0
        for n in range(1, 11):
            tmp = 1. + x[1] * n ** 2
            tmp = 0 if tmp < 0 else tmp
            f_n = x[0] * n * np.sqrt(tmp)
            for _i, value in enumerate(self.freq[_i:], start=_i):  # resume, where we just left for cpu time reasons
                if value > f_n:  # mask theoretical frequencies with inharmonicity
                    r += np.sum(self.amp[_i-2:_i+2])
                    break

        return -r
