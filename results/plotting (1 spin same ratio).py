import numpy as np
from numpy.random import default_rng
rng = default_rng()

import matplotlib.pyplot as plt
from tqdm import trange



avg1 = np.loadtxt("2D_1spin_data_avg(v1=1, tc=lc=10, N=1000).csv", delimiter=',')
var1 = np.loadtxt("2D_1spin_data_var(v1=1, tc=lc=10, N=1000).csv", delimiter=',')

avg2 = np.loadtxt("2D_1spin_data_avg(v1=1, tc=100, lc=1, N=1000).csv", delimiter=',')
var2 = np.loadtxt("2D_1spin_data_var(v1=1, tc=100, lc=1, N=1000).csv", delimiter=',')

avg3 = np.loadtxt("2D_1spin_data_avg(v1=1, tc=1, lc=100, N=1000).csv", delimiter=',')
var3 = np.loadtxt("2D_1spin_data_var(v1=1, tc=1, lc=100, N=1000).csv", delimiter=',')


totalTime = 10
timeStep = 0.1
t = np.arange(0, totalTime+timeStep, timeStep)

plt.plot(t, avg1, label=r"$\tau_c = l_c = 10$")
plt.fill_between(t, avg1+np.sqrt(var1), avg1-np.sqrt(var1), alpha=0.5)

plt.plot(t, avg2, label=r"$\tau_c = 100, l_c = 1$")
plt.fill_between(t, avg2+np.sqrt(var2), avg2-np.sqrt(var2), alpha=0.5)

plt.plot(t, avg3, label=r"$\tau_c = 1, l_c = 100$")
plt.fill_between(t, avg3+np.sqrt(var3), avg3-np.sqrt(var3), alpha=0.5)

plt.grid()
plt.legend()
plt.title("Dephasing coefficent of a single electron spin (2D)")
plt.ylabel(r"$\langle e^{i\phi} \rangle$")
plt.xlabel(r"Shuttling time $t_s$")
plt.show()



plt.semilogy(t, avg1, label=r"$\tau_c = l_c = 10$")
plt.fill_between(t, avg1+np.sqrt(var1), avg1-np.sqrt(var1), alpha=0.5)

plt.semilogy(t, avg2, label=r"$\tau_c = 100, l_c = 1$")
plt.fill_between(t, avg2+np.sqrt(var2), avg2-np.sqrt(var2), alpha=0.5)

plt.semilogy(t, avg3, label=r"$\tau_c = 1, l_c = 100$")
plt.fill_between(t, avg3+np.sqrt(var3), avg3-np.sqrt(var3), alpha=0.5)

plt.grid()
plt.legend()
plt.title("Dephasing coefficent of a single electron spin (2D)")
plt.ylabel(r"$\ln{\langle e^{i\phi} \rangle$}")
plt.ylim([2e0, 1e-3])
plt.xlabel(r"Shuttling time $t_s$")
plt.show()