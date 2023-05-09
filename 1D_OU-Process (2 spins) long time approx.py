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


def differenceRealization(totalTime, timeStep, gamma, delay):
    '''
    Function uses the transition probability found through the Fokker-Planck equation as well as
    the single point pdf.
    '''

    #Initialize arrays and necessary parameters
    totalSteps = int(totalTime/timeStep + 1) #+1 to include t=0
    t = np.arange(0, totalTime + timeStep, timeStep)

    #Generate an OU-process first
    B1 = OU_time_realization(totalTime, timeStep, gamma)

    #Using this generated realization, sample a new realization after the delay using the transition probability
    mean = B1 * np.exp(-gamma * delay)
    var = 1/(2*gamma) * (1-np.exp(-2*gamma*delay))

    B2 = rng.normal(mean, np.sqrt(var), totalSteps)

    difference = B1 - B2

    return difference



def functional(totalTime, timeStep, gamma, delay):
    '''
    Calculates the desired functional e^{i \int_0^t B(s) ds}, with B(s) being the OU-process.
    '''
    

    #Initialize arrays and necessary parameters
    totalSteps = int(totalTime/timeStep + 1) #+1 to include t=0
    functional = np.empty(totalSteps)

    #Generate realization of the difference process
    diff = differenceRealization(totalTime, timeStep, gamma, delay)

    integral = np.cumsum(diff * timeStep)
    integral[0] = 0
    functional = np.exp(1j * integral)


    return functional


def averaging(totalTime, timeStep, gamma, delay, N):
    '''
    Averages the functional of the two realizations B1 and B2 seperated in time with a defined delay.
    '''


    #Initalize arrays and necessary parameters
    totalSteps = int(totalTime/timeStep + 1) #+1 to include t=0
    samples = np.empty([N, totalSteps])
    t = np.arange(0, totalTime + timeStep, timeStep)

    avg = np.empty(totalSteps)
    var = np.empty(totalSteps)

    for i in trange(N):
        samples[i, :] = functional(totalTime, timeStep, gamma, delay)

    for i in range(totalSteps):
        slice = samples[:, i]
        avg[i] = np.mean(slice)
        var[i] = np.var(slice)/N

    return t, np.abs(avg), np.abs(var)


N = 1000
totalTime = 10
timeStep = 0.01
t = np.arange(0, totalTime + timeStep, timeStep)

#t, avg, var = averaging(totalTime, timeStep, gamma, delay, N)

#stddev = np.sqrt(var)
#err = stddev

#plt.plot(t, avg)
#plt.fill_between(t, avg+err, avg-err, alpha=0.5)

delay = np.array([0.001, 10, 100])
t_c = np.array([0.01, 10, 100])
gamma = 1/t_c


#a = np.sqrt(1/(2*gamma)) * delay
a = np.sqrt(1/(2*gamma)) * t_c * (1-np.exp(-gamma*delay))


t, avg1, var1 = averaging(totalTime, timeStep, gamma[0], delay[0], N)
#np.savetxt("1D_2spin_data_avg(tc=1).csv", avg1, delimiter=',')
#np.savetxt("1D_2spin_data_var(tc=1).csv", var1, delimiter=',')

#t, avg2, var2 = averaging(totalTime, timeStep, gamma[1], delay[1], N)
#np.savetxt("1D_2spin_data_avg(tc=10).csv", avg2, delimiter=',')
#np.savetxt("1D_2spin_data_var(tc=10).csv", var2, delimiter=',')

#t, avg3, var3 = averaging(totalTime, timeStep, gamma[2], delay[2], N)
#np.savetxt("1D_2spin_data_avg(tc=100).csv", avg1, delimiter=',')
#np.savetxt("1D_2spin_data_var(tc=100).csv", var1, delimiter=',')

err1 = np.sqrt(var1)
#err2 = np.sqrt(var2)
#err3 = np.sqrt(var3)

plt.plot(t, avg1, label=r"$T = $"+str(delay[0]))
#plt.plot(t, avg2, label=r"$T = $"+str(delay[1]))
#plt.plot(t, avg3, label=r"$T = $"+str(delay[2]))


analytical1 = np.exp(-a[0]**2 * (t/t_c[0] - 1 + np.exp(-t/t_c[0])))
#analytical2 = np.exp(-a[1]**2 * (t/t_c[1] - 1 + np.exp(-t/t_c[1])))
#analytical3 = np.exp(-a[2]**2 * (t/t_c[2] - 1 + np.exp(-t/t_c[2])))

tlarge = np.exp((-(1/(2*gamma[0]) * t_c[0] * t * (1-np.exp(-gamma[0]*delay[0])) + 1/(2*gamma[0])*t_c[0]**2)))

plt.plot(t, tlarge, label=r"Long Time approximation, $\tau_c =$"+str(t_c[0]), linestyle="dashed")

plt.plot(t, analytical1, label=r"Analytical solution $T = $"+str(delay[0]), linestyle="dashed")
#plt.plot(t, analytical2, label=r"Analytical solution $T = $"+str(delay[1]), linestyle="dashed")
#plt.plot(t, analytical3, label=r"Analytical solution $T = $"+str(delay[2]), linestyle="dashed")

plt.fill_between(t, avg1+err1, avg1-err1, alpha=0.5)
#plt.fill_between(t, avg2+err2, avg2-err2, alpha=0.5)
#plt.fill_between(t, avg3+err3, avg3-err3, alpha=0.5)

plt.ylabel(r"$\langle e^{i(\phi_1 - \phi_2)} \rangle$")
plt.ylim([0.9, 1])
plt.xlabel(r"Shuttling time $t_s$")
plt.legend()
plt.grid()
plt.title(r"Dephasing Coefficient of Two Spins with $\tau_c = $" + str(t_c[0]))
plt.show()
