import numpy as np
from numpy.random import default_rng
rng = default_rng()

import matplotlib.pyplot as plt
from tqdm import trange



def OU_time_realization(totalTime, timeStep, gamma):
    '''
    Function creating a realization of a stationary OU-process with mean = 0 and variance (1/(2*gamma)). This stationary process
    only exists for a certain initial condition of x(0): a normal distribution with mean 0 and variance (1/2*(gamma)). 

    totalTime defines the total process length, timeStep the stepsize, and gamma is the mean reversion rate.

    This implementation has slight issues due to the np.cumsum implementation: overflow errors occur. To fix this,
    one needs to combine the last two exponents in the last sum or find another solution.
    '''


    #Initializing arrays and necessary parameters
    totalSteps = int(totalTime/timeStep + 1) #+1 to include t=0
    t = np.arange(0, totalTime + timeStep, timeStep)
    ou = np.empty(totalSteps)

    #Defining Wiener increments:
    stddev = np.sqrt(timeStep)
    dW = rng.normal(0, stddev, totalSteps)

    #Stationary initial condition
    ou0 = rng.normal(0, np.sqrt(1/(2*gamma)))

    #Generating the realization of the OU-process
    ou = ou0 * np.exp(-gamma*t) + np.exp(-gamma * t) * np.cumsum(np.exp(gamma * t) * dW)


    return ou



def functional(totalTime, timeStep, gamma):
    '''
    Calculates the desired functional e^{i \int_0^t B(s) ds}, with B(s) being the OU-process.
    '''
    

    #Initialize arrays and necessary parameters
    totalSteps = int(totalTime/timeStep + 1) #+1 to include t=0
    t = np.arange(0, totalTime + timeStep, timeStep)
    functional = np.empty(totalSteps)

    #Generate realization of the OU-process
    B = OU_time_realization(totalTime, timeStep, gamma)

    integral = np.cumsum(B * timeStep)
    integral[0] = 0
    functional = np.exp(1j * integral)

    return functional


def averaging(totalTime, timeStep, gamma, N):
    '''
    Function samples the desired functional N times and averages over all realizations. It also calculates the variance 
    for the estimator of the average.
    '''

    #Initializing arrays
    totalSteps = int(totalTime/timeStep + 1)
    samples = np.empty([N, totalSteps])
    t = np.arange(0, totalTime + timeStep, timeStep)

    avg = np.empty(totalSteps)
    var = np.empty(totalSteps)


    for i in trange(N):
        samples[i, :] = functional(totalTime, timeStep, gamma)


    for i in range(totalSteps):
        slice = samples[:, i]
        avg[i] = np.mean(slice)
        var[i] = np.var(slice)/N


    return t, np.abs(avg), np.abs(var)


N = 1000000
totalTime = 10
timeStep = 0.01

t_c = 3
gamma = 1/t_c

t_c1 = 1
t_c2 = 3
t_c3 = 10

gamma1 = 1/t_c1
gamma2 = 1/t_c2
gamma3 = 1/t_c3

#t, avg, var = averaging(totalTime, timeStep, gamma, N)

#stddev = np.sqrt(var)
#err = stddev

t, avg1, var1 = averaging(totalTime, timeStep, gamma1, N)
t, avg2, var2 = averaging(totalTime, timeStep, gamma2, N)
t, avg3, var3 = averaging(totalTime, timeStep, gamma3, N)

err1 = np.sqrt(var1)
err2 = np.sqrt(var2)
err3 = np.sqrt(var3)


a1 = np.sqrt(1/(2*gamma1)) * t_c1
a2 = np.sqrt(1/(2*gamma2)) * t_c2
a3 = np.sqrt(1/(2*gamma3)) * t_c3

analytical1 = np.exp(-a1**2 * (t/t_c1 - 1 + np.exp(-t/t_c1)))
analytical2 = np.exp(-a2**2 * (t/t_c2 - 1 + np.exp(-t/t_c2)))
analytical3 = np.exp(-a3**2 * (t/t_c3 - 1 + np.exp(-t/t_c3)))


tlarge = np.exp((-(1/(2*gamma1) * t_c1 * t + 1/(2*gamma1)*t_c1**2)))


plt.semilogy(t, avg1, label=r"Simulation, $\tau_c = $"+str(t_c1)) #$\tau_c = $"+str(t_c1))
plt.fill_between(t, avg1+err1, avg1-err1, alpha=0.5)
plt.semilogy(t, analytical1, label=r"Analytical solution, $\tau_c =$"+str(t_c1), linestyle="dashed")
#plt.semilogy(t, tlarge, label=r"Analytical solution, $\tau_c =$"+str(t_c1), linestyle="dashed")

#np.savetxt("1D_1spin_data_avg(tc=1).csv", avg1, delimiter=',')
#np.savetxt("1D_1spin_data_var(tc=1).csv", var1, delimiter=',')

#plt.semilogy(t, tlarge, label=r"Analytical solution")

plt.semilogy(t, avg2, label=r"Simulation, $\tau_c = $"+str(t_c2))
plt.fill_between(t, avg2+err2, avg2-err2, alpha=0.5)
plt.semilogy(t, analytical2, label=r"Analytical solution, $\tau_c =$"+str(t_c2), linestyle="dashed")
#np.savetxt("1D_1spin_data_avg(tc=10).csv", avg2, delimiter=',')
#np.savetxt("1D_1spin_data_var(tc=10).csv", var2, delimiter=',')

plt.semilogy(t, avg3, label=r"Simulation, $\tau_c = $"+str(t_c3))
plt.fill_between(t, avg3+err3, avg3-err3, alpha=0.5)
plt.semilogy(t, analytical3, label=r"Analytical solution, $\tau_c =$"+str(t_c3), linestyle="dashed")
#np.savetxt("1D_1spin_data_avg(tc=100).csv", avg3, delimiter=',')
#np.savetxt("1D_1spin_data_var(tc=100).csv", var3, delimiter=',')



plt.ylabel(r"$\langle e^{i\phi} \rangle$")
plt.ylim([1e-3, 2e0])
plt.xlabel(r"Shuttling time $t_s$")
plt.title(r"Dephasing Coefficient of a Single Spin")
plt.grid()
plt.legend()
plt.show()



plt.plot(t, avg1, label=r"Simulation, $\tau_c = $"+str(t_c1))
plt.fill_between(t, avg1+err1, avg1-err1, alpha=0.5)
plt.plot(t, analytical1, label=r"Analytical solution, $\tau_c =$"+str(t_c1), linestyle="dashed")
#plt.plot(t, tlarge, label=r"Analytical solution, $\tau_c =$"+str(t_c1), linestyle="dashed")

#plt.plot(t, tlarge, label=r"Analytical solution")

plt.plot(t, avg2, label=r"Simulation, $\tau_c = $"+str(t_c2))
plt.fill_between(t, avg2+err2, avg2-err2, alpha=0.5)
plt.plot(t, analytical2, label=r"Analytical solution, $\tau_c =$"+str(t_c2), linestyle="dashed")

plt.plot(t, avg3, label=r"Simulation, $\tau_c = $"+str(t_c3))
plt.fill_between(t, avg3+err3, avg3-err3, alpha=0.5)
plt.plot(t, analytical3, label=r"Analytical solution, $\tau_c =$"+str(t_c3), linestyle="dashed")


plt.ylabel(r"$\langle e^{i\phi} \rangle$")
plt.xlabel(r"Shuttling time $t_s$")
plt.title(r"Dephasing Coefficient of a Single Spin")
plt.grid()
plt.legend()
plt.show()