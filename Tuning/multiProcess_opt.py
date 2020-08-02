from multiprocessing import Process, Queue
from scipy.optimize import minimize
import numpy as np


class threaded_opt:
    """
    driver for multiprocessing:
    called from harmonics to maximize cross correlation. In this case the two 1-dimensional sequences have been reduced
    to those elements not zero to speed up the code.
    Each minimization call is preformed by a discrete process (multiprocessing)
    We are still playing around with the optimization method, that sucks somehow
    """

    # initializations
    def __init__(self, amp, freq, ind, height):

        self.queue = Queue()
        self.function = self.chi_square
        self.amplitude = amp
        self.freq = freq
        self.a = ind
        # this is still a lousy solution, need to revise
        if len(ind) != 0:
            for i in range(2, 7):
                self.a = np.append(self.a, ind[0] / i)
            print(self.a)
        self.num_threads = len(self.a)
        self.best_fun = None
        self.best_x = None

        self.bnds = lambda x: ((0.95 * x, 1.05 * x), (0, 0.001))  # not needed with some optimizers

    # Run the optimization. Make the threads here.
    def run(self):

        processes = []
        for x in range(self.num_threads):  # Make the threads and start them off
            p = Process(target=self.target_function,
                        args=(x,))
            processes.append(p)
            p.start()

        self.best_fun = 1.e36
        for p in processes:
            chi, x = self.queue.get()
            # print(chi, x)
            if chi < self.best_fun:
                self.best_x = x
                self.best_fun = chi
        self.queue.close()
        self.queue.join_thread()
        for p in processes:
            p.join()

        return None

    # Each thread goes through this.
    def target_function(self, thread_ID):

        result = minimize(self.function,
                          [self.a[thread_ID], 0.],
                          bounds=self.bnds(self.a[thread_ID]),
                          method="Powell"  # "nelder-mead"
                         )
        self.queue.put((result.fun, result.x))

    # do the cross-correlation and maximize it (negate result for optimizer)
    def chi_square(self, x):
        r = 0
        _i = 0
        for n in range(1, 8):
            tmp = 1. + x[1] * n ** 2
            tmp = 0 if tmp < 0 else tmp
            f_n = x[0] * n * np.sqrt(tmp)
            for _i, value in enumerate(self.freq[_i:], start=_i):  # resume, where we just left for cpu time reasons
                if value > f_n:  # mask theoretical frequencies with inharmonicity
                    r += np.sum(self.amplitude[_i-2:_i+2])
                    break

        return -r
