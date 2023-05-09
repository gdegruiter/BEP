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


def plotRealization(totalTime, timeStep, gamma):

    totalSteps = int(totalTime/timeStep + 1) #+1 to include t=0
    t = np.arange(0, totalTime + timeStep, timeStep)

    plt.plot(t, OU_time_realization(totalTime, timeStep, gamma))
    plt.xlabel("Time")
    plt.ylabel("")
    plt.grid()
    plt.title("Realization of an OU-process")
    plt.show()
    return



def testingOnePoint(totalTime, timeStep, gamma, N, t1, t2, t3):
    '''
    Function samples N times the OU-process at certain points t, plotting a histogram of the found values.
    '''
    
    
    #Probability density function of the OU-process. The OU-process is stationary for a specific initial condition: x(0) ~ N(0, 1/(2*gamma)) (N(mean, variance))
    b = 1/(2*gamma)
    x = np.linspace(-20, 20, 100)
    pdf = 1/np.sqrt(2*np.pi * b) * np.exp(-(x)**2 / (2*b))

    #Sampling values at time = t for N realizations of the Ornstein-Uhlenbeck process.
    samples1 = np.empty(N)
    samples2 = np.empty(N)
    samples3 = np.empty(N)
    for i in trange(N):
        x0 = rng.normal(0, np.sqrt(b))
        process = OU_time_realization(totalTime, timeStep, gamma)
        samples1[i] = process[t1]
        samples2[i] = process[t2]
        samples3[i] = process[t3]

    plt.plot(x, pdf)
    plt.hist(samples1, bins=70, density=True, alpha=0.7, label="step: "+str(t1))
    plt.hist(samples2, bins=70, density=True, alpha=0.5, label="step: "+str(t2))
    plt.hist(samples2, bins=70, density=True, alpha=0.3, label="step: "+str(t3))
    plt.grid()
    plt.legend()
    plt.show()

    return


def testingCorrelation(totalTime, timeStep, gamma, N):
    '''
    Finds the autocorrelation function of the OU-process by averaging N samples.
    '''

    #Initializing arrays and necessary parameters
    totalSteps = int(totalTime/timeStep + 1) #+1 to include t=0
    t = np.arange(0, totalTime + timeStep, timeStep)

    corrSamples = np.empty([N, totalSteps])
    corr = np.empty(totalSteps)

    for i in trange(N):
        process = OU_time_realization(totalTime, timeStep, gamma)
        corrSamples[i, :] = process[0] * process

    for j in range(totalSteps):
        slice = np.mean(corrSamples[:, j])
        corr[j] = slice
    
    trueProcess = 1/(2*gamma) * np.exp(-gamma * t)
    
    plt.plot(t, corr, label="Simulated correlation")
    plt.plot(t, trueProcess, linestyle="dotted", label="Analytical correlation")
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation of the OU-process")
    plt.legend()
    plt.show()

    return

N = 100000

totalTime = 5
timeStep = 0.05

t_c = 100
gamma = 1/t_c


plotRealization(totalTime, timeStep, gamma)
#testingOnePoint(totalTime, timeStep, gamma, N, 0, 50, 100)
#testingCorrelation(totalTime, timeStep, gamma, N)