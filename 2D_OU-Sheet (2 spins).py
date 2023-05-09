import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from tqdm import trange
from numpy.random import default_rng
rng = default_rng()



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



def OU_sheet_realization(totalTime, timeStep, totalLength, xStep, tgamma, xgamma):
    '''
    Function creating a realization of an OU-sheet. Generating a time realization for each step of x, 
    using a certain solution to the 2D OU-process. 
    '''


    #Initializing arrays and necessary parameters
    totaltSteps = int(totalTime/timeStep + 1) #include t=0
    t = np.arange(0, totalTime + timeStep, timeStep)
    totalxSteps = int(totalLength/xStep + 1)
    x = np.arange(0, totalLength + xStep, xStep)

    sheet = np.empty([totalxSteps, totaltSteps])

    #Initialize sheet
    sheet = np.empty([totalxSteps, totaltSteps])
    xsheet = np.empty([totalxSteps, totaltSteps])
    tsheet = np.empty([totalxSteps, totaltSteps])
    
    sheet00 = rng.normal(0, np.sqrt(1/(4*xgamma*tgamma)))
    x0 = rng.normal(0, 1/(2*xgamma))
    
    for i in range(totalxSteps):
        tsheet[i,:] = OU_time_realization(totalTime, timeStep, tgamma)
    
    for j in range(totaltSteps):
        sheet[:, j] = sheet00 * np.exp(-xgamma * x) * np.exp(-tgamma * t[j]) + xStep * np.exp(-xgamma * x) * np.cumsum(tsheet[:, j] * np.exp(xgamma * x))

            
    return sheet



def dephasing_2spins(totalTime, timeStep, totalLength, xStep, tgamma, xgamma, v1, v2, T1, T2, delay):


    if v1[5] != v2[5]:
        raise ValueError("v1 = v2, different speeds not yet supported")
    #Initializing arrays
    t = np.arange(0, totalTime+timeStep, timeStep)
    x = np.arange(0, totalLength + xStep, xStep)


    sheet = OU_sheet_realization(totalTime, timeStep, totalLength, xStep, tgamma, xgamma)
    interpolation = interpolate.RectBivariateSpline(t, x, sheet)
    
    #Define angles
    a = np.arctan(v1)
    #a1 = np.arctan(v1)
    #a2 = np.arctan(v2)
    
    #Integration resolution
    sStep = 0.01
    S1 = np.sqrt(T1**2 * (1 + v1**2))
    S1[0] = 0 #Resolve infinite case for velocity = 0

    S2 = np.sqrt(T2**2 * (1 + v2**2))
    S2[0] = 0 #Resolve infinite case for velocity = 0

    dephase = np.empty(len(a))
    for i in range(len(a)):
        s1 = np.arange(0, S1[i] + sStep, sStep)
        s2 = np.arange(0, S2[i] + sStep, sStep)

        t1s = np.cos(a[i])*s1
        x1s = np.sin(a[i])*s1

        t2s = np.cos(a[i])*s2 + delay
        x2s = np.sin(a[i])*s2

        sheet1_s = interpolation.ev(t1s, x1s)
        sheet2_s = interpolation.ev(t2s, x2s)
    
        integral1 = np.sum(sheet1_s * sStep)
        integral2 = np.sum(sheet2_s * sStep)

        dephase[i] = np.exp(1j * (integral1 - integral2))

    dephase[0] = 0
    return dephase


def average_dephasing(totalTime, timeStep, totalLength, xStep, tgamma, xgamma, v1, v2, T1, T2, delay, N):
    
    #check sstep and stime to be the same as defined in the dephasing function
    totalsSteps = totalvSteps
    samples = np.empty([N, totalsSteps])
    avg = np.empty(totalsSteps)
    var = np.empty(totalsSteps)

    for i in trange(N):
        samples[i, :] = dephasing_2spins(totalTime, timeStep, totalLength, xStep, tgamma, xgamma, v1, v2, T1, T2, delay)

    for i in range(totalsSteps):
        slice = samples[:, i]
        avg[i] = np.mean(slice)
        var[i] = np.var(slice)/N

    return avg, var



shuttleLength = 3
vStep = 0.1
vMax = 5


v1 = np.arange(0, vMax + vStep, vStep)
v2 = v1
totalvSteps = len(v1)

T1 = shuttleLength / v1
T2 = shuttleLength / v2


totalTime = 15
timeStep = 0.05
t = np.arange(0, totalTime+timeStep, timeStep)

totalLength = 15
xStep = 0.05
x = np.arange(0, totalLength + xStep, xStep)


l_c1 = 1
xgamma1 = 1/l_c1
t_c1 = 10
tgamma1 = 1/t_c1

l_c2 = 10
xgamma2 = 1/l_c2
t_c2 = 10
tgamma2 = 1/t_c2

l_c3 = 10
xgamma3 = 1/l_c3
t_c3 = 1
tgamma3 = 1/t_c3

delay = 1

N = 1000


avg1, var1 = average_dephasing(totalTime, timeStep, totalLength, xStep, tgamma1, xgamma1, v1, v2, T1, T2, delay, N)
#np.savetxt("2D_2spins_data_avg.csv", avg1, delimiter=',')
#np.savetxt("2D_2spins_data_var.csv", var1, delimiter=',')

avg2, var2 = average_dephasing(totalTime, timeStep, totalLength, xStep, tgamma2, xgamma2, v1, v2, T1, T2, delay, N)
avg3, var3 = average_dephasing(totalTime, timeStep, totalLength, xStep, tgamma3, xgamma3, v1, v2, T1, T2, delay, N)

plt.plot(v1, avg1, label=r"$\tau_c = $" + str(t_c1) + r"$, l_c = $" + str(l_c1))
plt.plot(v1, avg2, label=r"$\tau_c = $" + str(t_c2) + r"$, l_c = $" + str(l_c2))
plt.plot(v1, avg3, label=r"$\tau_c = $" + str(t_c3) + r"$, l_c = $" + str(l_c3))

plt.fill_between(v1, avg1+np.sqrt(var1), avg1-np.sqrt(var1), alpha=0.5)
plt.fill_between(v1, avg2+np.sqrt(var2), avg2-np.sqrt(var2), alpha=0.5)
plt.fill_between(v1, avg3+np.sqrt(var3), avg3-np.sqrt(var3), alpha=0.5)

#plt.title(r"Dephasing coefficient of two spins with $\tau_c = $" + str(t_c) + r"$, l_c = $" + str(l_c))
plt.title(r"Dephasing coefficient of two spins with delay T = " + str(delay))
plt.xlabel(r"Shuttling velocity $v$")
plt.ylabel(r"$\langle e^{i (\phi_1 - \phi_2)} \rangle$")
plt.grid()
plt.legend()
plt.show()
