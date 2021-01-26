from multiprocessing import Process, Queue
from scipy.optimize import minimize, shgo, basinhopping
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


class MyBounds(object):

    def __init__(self, f0):
        self.xmax = np.array([1.005 * f0, INHARM])
        self.xmin = np.array([0.995 * f0, 0])

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))

        return tmax and tmin


class MyTakeStep(object):

    def __init__(self, initial):
        self.s0 = initial[0] * 0.002
        self.s1 = initial[1]
        self.first = True

    def __call__(self, x):
        # the very first call try local variables, thereafter global within range
        if self.first:
            self.first = False
        else:
            # stepsize for f0 within .2% of the initial value i.e. equal to 3.5 cent
            x[0] += np.random.uniform(-self.s0, self.s0)
            # stepsize for B is between factor .2 and 5 of the initial value unless B is zero in which case it is
            # between 0 and INHARM/2
            if self.s1 == 0 or x[1] <= 0:
                x[1] = 10**np.random.uniform(-6, np.log10(INHARM/2)) - 1.e-6
            else:
                x[1] *= 10**np.random.uniform(-.7, .7)

        return x


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

        for thread_id in range(self.num_threads):  # Make the threads and start them off
            p = Process(target=self.target_function,
                        args=(queue, thread_id,))
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
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
        """
        """
        result = minimize(self.chi_square,
                          [self.a[thread_id], self.b[thread_id]],
                          #bounds=bnds(self.a[thread_id]),
                          method="COBYLA",
                          constraints=cons(self.a[thread_id])
                          )        
        result = shgo(self.chi_square,
                      bounds=bnds(self.a[thread_id]),
                      sampling_method='sobol'
                      )
        """
        minimizer_kwargs = {"method": "L-BFGS-B",
                            "args": thread_id}
        mybounds = MyBounds(f0=self.a[thread_id])
        mytakestep = MyTakeStep(initial=[self.a[thread_id], self.b[thread_id]])
        result = basinhopping(self.chi_square,
                              [self.a[thread_id], self.b[thread_id]],
                              minimizer_kwargs=minimizer_kwargs,
                              niter=11,  # gets really time consuming for higher niter
                              accept_test=mybounds,
                              take_step=mytakestep
                              )

        queue.put((result.fun, result.x))

    # maximize the cross-correlation (negate result for minimizer)
    def chi_square(self, x, args):

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
