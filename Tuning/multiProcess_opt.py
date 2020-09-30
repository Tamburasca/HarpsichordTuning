from multiprocessing import Process, Queue
from scipy.optimize import minimize
import numpy as np
from operator import itemgetter
import logging
from .parameters import _debug, INHARM


format = "%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s"
logging.basicConfig(format=format,
                    level=logging.INFO,
                    datefmt="%H:%M:%S")
if _debug:
    logging.getLogger().setLevel(logging.DEBUG)


class ThreadedOpt:
    """
    driver for multiprocessing:
    called from harmonics to maximize cross correlation. In this case the two
    1-dimensional sequences have been reduced to those elements not zero to
    speed up the code.
    Each minimization call is preformed by a discrete process (multiprocessing)
    We are still playing around with the optimization method, that sucks somehow
    """

    # initializations
    def __init__(self, amp, freq, initial, height=None):

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

        queue = Queue()
        processes = []
        logging.debug("Number of processes: " + str(self.num_threads))

        for x in range(self.num_threads):  # Make the threads and start them off
            p = Process(target=self.target_function,
                        args=(queue, x,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=1)  # resume after timeout

        self.best_fun = 1.e36
        for _ in processes:
            chi, x = queue.get()
            logging.debug("chi, x: " + str(chi) + ',' + str(x))
            if chi < self.best_fun:
                self.best_x = x
                self.best_fun = chi
        queue.close()
        queue.join_thread()

        # clean up processes that did not join within timeout period
        while processes:
            p = processes.pop()
            if p.is_alive():
                p.terminate()
                logging.warning("Multiprocessing: Process did not join, cleaning up ...")

        return None

    # Each thread goes through this.
    def target_function(self, queue, thread_id):

        # set the boundaries for f_0 within .5% and b < INHARM
        def bnds(f0):
            return (0.995 * f0, 1.005 * f0), (0, INHARM)

        # since some methods do not accept boundaries, we convert them to constraints, harhar
        def cons(f0):
            bounds = np.asarray(bnds(f0))
            constraint = []
            for index in range(len(bounds)):
                lower, upper = bounds[index]
                # lower constraint first, upper second
                constraint.append({'type': 'ineq', 'fun': lambda x, lb=lower, i=index: x[i] - lb})
                constraint.append({'type': 'ineq', 'fun': lambda x, ub=upper, i=index: ub - x[i]})
            return constraint

        """
        see also here to follow minimization issues in more detail
        https://scipy-lectures.org/advanced/mathematical_optimization/
        http://people.duke.edu/~ccc14/sta-663-2016/13_Optimization.html
        https://stackoverflow.com/questions/12781622/does-scipys-minimize-function-with-method-cobyla-accept-bounds
        """
        result = minimize(self.chi_square,
                          [self.a[thread_id], self.b[thread_id]],
                          #bounds=bnds(self.a[thread_id]),
                          method="COBYLA",
                          constraints=cons(self.a[thread_id])
                          #options = {'eps': 1}
                          )
        queue.put((result.fun, result.x))

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
