import numpy as np
from numpy.random import default_rng
rng = default_rng()

import matplotlib.pyplot as plt
from tqdm import trange

totalTime = 5
timeStep = 0.1
t = np.arange(0, totalTime + timeStep, timeStep)

delay = 5

t_c1 = 1
gamma1 = 1/t_c1

a1 = np.sqrt(1/(2*gamma1)) * t_c1 #* (1-np.exp(-gamma1*delay))
analytical1 = np.exp(-a1**2 * (t/t_c1 - 1 + np.exp(-t/t_c1)))


t_c2 = 5
gamma2 = 1/t_c2

a2 = np.sqrt(1/(2*gamma2)) * t_c2 #* (1-np.exp(-gamma2*delay))
analytical2 = np.exp(-a2**2 * (t/t_c2 - 1 + np.exp(-t/t_c2)))


t_c3 = 10
gamma3= 1/t_c3

a3 = np.sqrt(1/(2*gamma3)) * t_c3 #* (1-np.exp(-gamma3*delay))
analytical3 = np.exp(-a3**2 * (t/t_c3 - 1 + np.exp(-t/t_c3)))


plt.plot(t, analytical1, label=r"$\tau_c$ = "+str(t_c1))
plt.plot(t, analytical2, label=r"$\tau_c$ = "+str(t_c2))
plt.plot(t, analytical3, label=r"$\tau_c$ = "+str(t_c3))

plt.ylabel(r"$\langle e^{i\phi} \rangle$")
plt.xlabel("t")
plt.title(r"$\langle e^{i\phi(t)} \rangle$ for different values of $\tau_c$")
plt.grid()
plt.legend()
plt.show()

plt.semilogy(t, analytical1, label=r"$\tau_c$ = "+str(t_c1))
plt.semilogy(t, analytical2, label=r"$\tau_c$ = "+str(t_c2))
plt.semilogy(t, analytical3, label=r"$\tau_c$ = "+str(t_c3))

plt.ylabel(r"$\langle e^{i\phi} \rangle}$")
plt.ylim([1e-2, 2e0])
plt.xlabel("t")
plt.title(r"$\langle e^{i\phi(t)} \rangle$ for different values of $\tau_c$")
plt.grid()
plt.legend()
plt.show()