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



def dephasing_1spin(totalTime, timeStep, totalLength, xStep, tgamma, xgamma, v1, T1):

    #Initializing arrays
    t = np.arange(0, totalTime+timeStep, timeStep)
    x = np.arange(0, totalLength + xStep, xStep)

    #X,T = np.meshgrid(x, t)

    sheet = OU_sheet_realization(totalTime, timeStep, totalLength, xStep, tgamma, xgamma)
    #XT = np.stack([X.ravel(), T.ravel()]).T
    #Interpolate the sheet
    interpolation = interpolate.RectBivariateSpline(t, x, sheet)
    
    #Calculate angle for coordinate transformation
    a = np.arctan(v1)

    #Integration resolution
    sStep = 0.01
    S1 = np.sqrt(T1**2 * (1 + v1**2))
    S1[0] = 0 #Resolve infinite case for velocity = 0

    dephase = np.empty(len(a))
    for i in range(len(a)):
        s = np.arange(0, S1[i] + sStep, sStep)

        t1s = np.cos(a[i])*s
        x1s = np.sin(a[i])*s

        #t1s_x1s = np.stack([t1s, x1s]).T
        sheet_s = interpolation.ev(t1s, x1s)
        integral = np.sum(sheet_s * sStep)
        dephase[i] = np.exp(1j * integral)

        dephase[0] = 0
    return dephase



def average_dephasing(totalTime, timeStep, totalLength, xStep, tgamma, xgamma, v1, T1, N):
    

    totalsSteps = totalvSteps #int(sTime / sStep + 1)
    samples = np.empty([N, totalsSteps])
    avg = np.empty(totalsSteps)
    var = np.empty(totalsSteps)

    for i in trange(N):
        samples[i, :] = dephasing_1spin(totalTime, timeStep, totalLength, xStep, tgamma, xgamma, v1, T1)

    for i in range(totalsSteps):
        slice = samples[:, i]
        avg[i] = np.mean(slice)
        var[i] = np.var(slice)/N

    return avg, var



shuttleLength = 3
vStep = 0.01
vMax = 5
v1 = np.arange(0, vMax + vStep, vStep)
totalvSteps = len(v1)

T1 = shuttleLength / v1

totalTime = 10
timeStep = 0.1
t = np.arange(0, totalTime+timeStep, timeStep)

totalLength = 10
xStep = 0.1
x = np.arange(0, totalLength + xStep, xStep)


l_c = 10
xgamma = 1/l_c

t_c = 10
tgamma = 1/t_c

N = 1000

avg1, var1 = average_dephasing(totalTime, timeStep, totalLength, xStep, 1, 1/50, v1, T1, N)
avg2, var2 = average_dephasing(totalTime, timeStep, totalLength, xStep, 1/2, 1/2, v1, T1, N)
avg3, var3 = average_dephasing(totalTime, timeStep, totalLength, xStep, 1/5, 1/5, v1, T1, N)
#np.savetxt("2D_1spin_data_avg.csv", avg, delimiter=',')
#np.savetxt("2D_1spin_data_var.csv", var, delimiter=',')

plt.plot(v1, avg1, label=r"$\tau_c, l_c = 1$")
plt.fill_between(v1, avg1+np.sqrt(var1), avg1-np.sqrt(var1), alpha=0.5)

plt.plot(v1, avg2, label=r"$\tau_c, l_c = 2$")
plt.fill_between(v1, avg2+np.sqrt(var2), avg2-np.sqrt(var2), alpha=0.5)

plt.plot(v1, avg3, label=r"$\tau_c, l_c = 5$")
plt.fill_between(v1, avg3+np.sqrt(var3), avg3-np.sqrt(var3), alpha=0.5)


plt.grid()
plt.title("Dephasing coefficent of a single electron spin (2D)")
plt.ylabel(r"$\langle e^{i\phi} \rangle$")
plt.xlabel(r"Shuttling velocity $v$")
plt.legend()
plt.show()
