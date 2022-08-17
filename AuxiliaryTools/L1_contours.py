from numpy import log10, arange, meshgrid, stack, array, amin, amax, append
import matplotlib.pyplot as plt
from matplotlib import cm
from Tuning.L1costfunction import L1
from Tuning.L2costfunction import L2

theCM = cm.get_cmap()

path = array([
    [204.92706698586852, 2.4512644719649316e-06],
    [205.20844629053195, 2.2518024050588937e-06],
    [205.86656605088206, 2.2518011764841346e-06],
    [205.32801030344967, 2.2518011764843057e-06],
    [205.32167213453704, 2.2518011764841346e-06],
    [205.30797902444158, 2.2518011764857304e-06],
])

# exact to b=.0001
# founds = [415.0, 830.124478217012, 1245.4978506647046, 1661.2444090805334,
#           2077.488259113232, 2494.353258899385, 2911.9629581560675,
#           3330.4405379142177, 3749.908751014604, 4170.489863484847]

founds = [(615.0377859618212, 3), (820.3618099250172, 4), (1026.4150561079955, 5), (1641.7718206380123, 8),
          (1847.5854276270459, 9), (2053.334925271951, 10), (2258.9719468724015, 11), (2464.297240840511, 12),
          (2875.4735685955284, 14), (3082.9471642076783, 15), (3289.285650695724, 16), (3493.6590575883847, 17),
          (3701.5213559994536, 18)]

l1_min = L1(founds)

initial = [204.971, 4.504e-05, 0.019602835461558023]
final = [205.138, 1.511e-05, 0.003178838204573313]

# define grid
# lower, upper limit f0
# xm_l, xm_u = 0.995, 1.005
xm_u = max(initial[0], final[0], amax(path, axis=0)[0]) * 1.003
xm_l = min(initial[0], final[0], amin(path, axis=0)[0]) * 0.997
x = arange(xm_l,
           xm_u,
           (xm_u - xm_l) / 500.)
# lower, upper limit B
# ym_l, ym_u = -5.0, -3.7
ym_u = max(log10(initial[1]), log10(final[1]), log10(amax(path, axis=0))[1]) + .1
ym_l = min(log10(initial[1]), log10(final[1]), log10(amin(path, axis=0))[1]) - .1
y = arange(ym_l,
           ym_u,
           (ym_u - ym_l) / 500.)
xgrid, ygrid = meshgrid(x, y)
xy = stack([xgrid, ygrid])

res_l1 = list()
res_jac_f0 = list()
res_jac_b = list()

for y_data in y:
    tmp_l1 = list()
    tmp_jac_f0 = list()
    tmp_jac_b = list()

    for x_data in x:
        tmp = l1_min.l1_minimum_der([x_data, 10 ** y_data])
        tmp_l1.append(tmp[0])
        tmp_jac_f0.append(tmp[1][0])
        tmp_jac_b.append(tmp[1][1])
    res_l1.append(tmp_l1)
    res_jac_f0.append(tmp_jac_f0)
    res_jac_b.append(tmp_jac_b)

res_l1 = array(res_l1)
res_jac_f0 = array(res_jac_f0)
res_jac_b = array(res_jac_b)

plt.figure(1)
cs = plt.contour(xgrid, ygrid, res_l1, 25, cmap='hsv')
plt.clabel(cs, inline=True, fontsize=10)
plt.title("L1 cost function")
plt.xlabel("f0/Hz")
plt.ylabel("B (log)")
plt.scatter(initial[0], log10(initial[1]), c='orange', label="initial")
plt.scatter(final[0], log10(final[1]), c='black', label="final")
plt.scatter(path.T[0], log10(path.T[1]))
plt.plot(
    append(append(array(initial[0]), path.T[0]), final[0]),
    append(append(array(log10(initial[1])), log10(path.T[1])), log10(final[1]))
)
plt.legend()

plt.figure(2)
cs = plt.contour(xgrid, ygrid, res_jac_f0, 30, cmap='hsv')
plt.clabel(cs, inline=True, fontsize=10)
plt.title("L1 cost function - Jacobian derived to f0")
plt.xlabel("f0/Hz")
plt.ylabel("B (log)")
plt.scatter(initial[0], log10(initial[1]), c='orange', label="initial")
plt.scatter(final[0], log10(final[1]), c='black', label="final")
plt.scatter(path.T[0], log10(path.T[1]))
plt.plot(
    append(append(array(initial[0]), path.T[0]), final[0]),
    append(append(array(log10(initial[1])), log10(path.T[1])), log10(final[1]))
)
plt.legend()

plt.figure(3)
cs = plt.contour(xgrid, ygrid, res_jac_b, 20, cmap='hsv')
plt.clabel(cs, inline=True, fontsize=10)
plt.title("L1 cost function - Jacobian derived to B")
plt.xlabel("f0/Hz")
plt.ylabel("B (log)")
plt.scatter(initial[0], log10(initial[1]), c='orange', label="initial")
plt.scatter(final[0], log10(final[1]), c='black', label="final")
plt.scatter(path.T[0], log10(path.T[1]))
plt.plot(
    append(append(array(initial[0]), path.T[0]), final[0]),
    append(append(array(log10(initial[1])), log10(path.T[1])), log10(final[1]))
)
plt.legend()

plt.show()
