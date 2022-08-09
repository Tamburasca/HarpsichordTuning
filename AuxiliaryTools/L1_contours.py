from numpy import log10, arange, meshgrid, stack, array
import matplotlib.pyplot as plt
from matplotlib import cm
from Tuning.FFTL1minimizer import L1
# from Tuning.FFTL2minimizer import L2

theCM = cm.get_cmap()

path = array([
    [204.7966731750672, 4.2116798259568846e-05],
    [204.7833563582234, 3.329745185019745e-05],
    [204.79230324922423, 2.751359512261169e-05],
    [204.82792659445047, 2.5275328028849077e-05],
    [205.02883815558795, 1.9116356949792753e-05],
    [205.06643342802082, 1.796386878877137e-05],
    [205.11458857956003, 1.332446401156638e-05],
    [205.12997425235338, 1.286556449919784e-05],
    [205.1306063681419, 1.2868091174278664e-05],
    [205.12239889606863, 1.3716253325820311e-05],
    [205.1213798848879, 1.3827502240565212e-05],
    [205.12136811465615, 1.3830511300352465e-05],
    [205.12136829740368, 1.383059865472359e-05],
    [205.12136830862187, 1.383060401707784e-05],
    [205.12136830917808, 1.3830604282951004e-05],
])

# exact to b=.0001
# founds = [415.0, 830.124478217012, 1245.4978506647046, 1661.2444090805334,
#           2077.488259113232, 2494.353258899385, 2911.9629581560675,
#           3330.4405379142177, 3749.908751014604, 4170.489863484847]

founds = [410.0558671479703, 615.417157658387, 820.5295023411812, 1025.9443956322136, 1641.4065838519218,
          1847.1366546062698, 2052.623065874455, 2258.5381074926077, 2463.4877647973153, 3081.6041632989713,
          3288.056664857722, 3493.1814390134696]

# founds = [400., 802., 1204., 1607., 2011.]

l1_min = L1(founds)

initial = [204.997, 7.521e-05, 99.71984185974736]
final = [205.121, 1.383e-05, 2.617694814346919]
# define grid
# lower, upper limit f0
xm_l, xm_u = 0.997, 1.003
x = final[0] * arange(xm_l,
                      xm_u,
                      (xm_u - xm_l) / 500.)
# lower, upper limit B
ym_l, ym_u = -5.0, -3.7
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

fig = plt.figure(1)
cs = plt.contour(xgrid, ygrid, res_l1, 20, cmap='hsv')
plt.clabel(cs, inline=True, fontsize=10)
plt.title("L1 cost function")
plt.xlabel("f0/Hz")
plt.ylabel("B (log)")
plt.scatter(initial[0], log10(initial[1]), label="initial")
plt.scatter(final[0], log10(final[1]), label="final")
plt.scatter(path.T[0], log10(path.T[1]), label="path")
plt.legend()

fig = plt.figure(2)
cs = plt.contour(xgrid, ygrid, res_jac_f0, 14, cmap='hsv')
plt.clabel(cs, inline=True, fontsize=10)
plt.title("L1 cost function - Jacobian derived to f0")
plt.xlabel("f0/Hz")
plt.ylabel("B (log)")
plt.scatter(initial[0], log10(initial[1]), label="initial")
plt.scatter(final[0], log10(final[1]), label="final")
plt.scatter(path.T[0], log10(path.T[1]), label="path")
plt.legend()

fig = plt.figure(3)
cs = plt.contour(xgrid, ygrid, res_jac_b, 14, cmap='hsv')
plt.clabel(cs, inline=True, fontsize=10)
plt.title("L1 cost function - Jacobian derived to B")
plt.xlabel("f0/Hz")
plt.ylabel("B (log)")
plt.scatter(initial[0], log10(initial[1]), label="initial")
plt.scatter(final[0], log10(final[1]), label="final")
plt.scatter(path.T[0], log10(path.T[1]), label="path")
plt.legend()

plt.show()
