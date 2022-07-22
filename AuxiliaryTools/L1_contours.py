import numpy as np
from numdifftools import Jacobian, Hessian
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D

theCM = cm.get_cmap()


def bisection(array, value):
    """
    Given an <array>, and given a <value>, returns an index j such that
    <value< is between array[j] and array[j+1]. <array> must increase
    monotonically. j=-1 or j=len(array) is returned to indicate that
    <value> is out of range below and above, respectively.
    ref.:
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    n = len(array)
    if value < array[0]:
        return -1
    elif value > array[n-1]:
        return n
    jl = 0  # Initialize lower
    ju = n - 1  # and upper limits.
    while ju-jl > 1:
        # If we are not yet done,
        jm = (ju+jl) >> 1  # compute a midpoint with a bitshift
        if value >= array[jm]:
            jl = jm  # and replace either the lower limit
        else:
            ju = jm  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if value == array[0]:  # edge cases at bottom
        return 0
    elif value == array[n-1]:  # and top
        return n-1
    else:
        return jl


class L1(object):
    def __init__(self, ind):
        """
        :param ind: array - measured resonance frequencies as from peaks (FFT)
        """
        self.fo = ind
        self.fmax = max(self.fo)
        self.l1_first = None
        self.l1_last = None
        self.jacobi = np.array([0., 0.])

    def l1_minimum(self, x0, jac: bool = False):
        """
        returns the cost function for a regression on the L1 norm
        l1 = sum( abs(f_i(measured) - f_i(calculated) ) )
        :param x0: array - [f0, b] such that
            f = i * x0[0] * sqrt(1. + x0[1] * i**2)
        :param jac: bool - if jacobi is to be calculated
        :return: float - l1 cost function
        """
        freq = list()
        x0 = list(x0)  # x0 comes as ndarray if invoked from the scipy minimizer
        self.jacobi = np.array([0., 0.])

        for i in range(1, 640):
            f = i * x0[0] * np.sqrt(1. + x0[1] * i * i)
            freq.append(f)
            if f > self.fmax:
                break  # exit if superseded max. frequency measured to save time
        num_freq = len(freq)
        l1 = 0.  # l1 cost function
        for found in self.fo:
            # loop over peaks found
            idx = bisection(freq, found)
            # L1 norm
            if idx == -1:
                diff = found - freq[0]
                l1 += abs(diff)  # <min frequency
                if jac:
                    self.jacobi += \
                        self.__derivative(x0=x0, i=idx) * np.sign(-diff)
            elif idx == num_freq:
                diff = found - freq[num_freq - 1]
                l1 += abs(diff)  # >max frequency
                if jac:
                    self.jacobi += \
                        self.__derivative(x0=x0, i=idx) * np.sign(-diff)
            else:
                # consider the closest candidate of neighbors
                l1 += min(abs(found - freq[idx]), abs(found - freq[idx + 1]))
                if jac:
                    if abs(found - freq[idx]) < abs(found - freq[idx + 1]):
                        diff = found - freq[idx]
                        self.jacobi += \
                            self.__derivative(x0=x0, i=idx) * np.sign(-diff)
                    else:
                        diff = found - freq[idx + 1]
                        self.jacobi += \
                            self.__derivative(x0=x0, i=idx + 1) * np.sign(-diff)

        if self.l1_first is None:
            self.l1_first = l1
        self.l1_last = l1
        # print(x0, l1, self.jacobi)
        return l1

    def l1_minimum_jac(self, x0):
        return Jacobian(lambda x0: self.l1_minimum(x0))(x0).ravel()

    def l1_minimum_hess(self, x0):
        return Hessian(lambda x0: self.l1_minimum(x0))(x0)

    def l1_minimum_der(self, x0):
        # Jacobian is second parameter derived from derivations
        return self.l1_minimum(x0, jac=True), self.jacobi

    def compare_l1(self) -> bool:
        return self.l1_last < self.l1_first

    @staticmethod
    def __derivative(x0, i):
        if x0[1] < 0.:
            x0[1] = 0.
        deriv_f0 = i * np.sqrt(1. + x0[1] * i * i)
        deriv_b = 0.5 * i ** 3 * x0[0] / np.sqrt(1. + x0[1] * i * i)
        return np.array([deriv_f0, deriv_b])



# exact to b=.0001
# founds = [415.0, 830.124478217012, 1245.4978506647046, 1661.2444090805334,
#          2077.488259113232, 2494.353258899385, 2911.9629581560675,
#          3330.4405379142177, 3749.908751014604, 4170.489863484847]

founds = [410.04215516269574, 615.0121045459148, 820.5319323791974,
          1025.8776214323211, 1641.5581842569156, 1847.2528326482418,
          2052.875537825593, 2258.691350329363, 2463.506321941871,
          2875.1564428260713, 3082.953419636512, 3288.791207301226,
          3493.3723146015595, 4941.389842345284, 6190.50897725149,
          6400.746636886079]

# founds = [400., 802., 1204., 1607., 2011.]

l1_min = L1(founds)

initial = [205.006, 4.121e-05]
final = [205.111, 1.347e-05]
# define grid
x = np.arange(0.998*initial[0],
              1.002*initial[0],
              .0025)
y = np.arange(
    -5.5,
    -4.5,
    .00125)
xgrid, ygrid = np.meshgrid(x, y)
xy = np.stack([xgrid, ygrid])

res_l1 = list()
res_jac_f0 = list()
res_jac_b = list()

for y_data in y:
    tmp_l1 = list()
    tmp_jac_f0 = list()
    tmp_jac_b = list()

    for x_data in x:
        tmp = l1_min.l1_minimum_der([x_data, 10**y_data])
        tmp_l1.append(tmp[0])
        tmp_jac_f0.append(tmp[1][0])
        tmp_jac_b.append(tmp[1][1])
    res_l1.append(tmp_l1)
    res_jac_f0.append(tmp_jac_f0)
    res_jac_b.append(tmp_jac_b)

res_l1 = np.array(res_l1)
res_jac_f0 = np.array(res_jac_f0)
res_jac_b = np.array(res_jac_b)

fig = plt.figure(1)
cs = plt.contour(xgrid, ygrid, res_l1, 20, cmap='hsv')
plt.clabel(cs, inline=True, fontsize=10)
plt.title("L1 cost function")
plt.xlabel("f0/Hz")
plt.ylabel("B (log)")
plt.scatter(initial[0], np.log10(initial[1]), label="initial")
plt.scatter(final[0], np.log10(final[1]), label="final")
plt.legend()

fig = plt.figure(2)
cs = plt.contour(xgrid, ygrid, res_jac_f0, 14, cmap='hsv')
plt.clabel(cs, inline=True, fontsize=10)
plt.title("L1 cost function - Jacobian derived to f0")
plt.xlabel("f0/Hz")
plt.ylabel("B (log)")
plt.scatter(initial[0], np.log10(initial[1]), label="initial")
plt.scatter(final[0], np.log10(final[1]), label="final")
plt.legend()

fig = plt.figure(3)
cs = plt.contour(xgrid, ygrid, res_jac_b, 14, cmap='hsv')
plt.clabel(cs, inline=True, fontsize=10)
plt.title("L1 cost function - Jacobian derived to B")
plt.xlabel("f0/Hz")
plt.ylabel("B (log)")
plt.scatter(initial[0], np.log10(initial[1]), label="initial")
plt.scatter(final[0], np.log10(final[1]), label="final")
plt.legend()

plt.show()