import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
from numpy.random import default_rng
rng = default_rng()


def OU_time_realization(totalTime, timeStep, gamma):
    '''
    Function creating a realization of a stationary OU-process with mean = 0 and variance (1/(2*gamma)). This stationary process
    only exists for a certain initial condition of x(0): a normal distribution with mean 0 and variance (1/2*(gamma)). 

    totalTime defines the total process length, timeStep the stepsize, and gamma is the mean reversion rate.
    '''


    #Initializing arrays and necessary parameters
    totalSteps = int(totalTime/timeStep + 1) #+1 to include t=0
    t = np.arange(0, totalTime + timeStep, timeStep)
    ou = np.empty(totalSteps)

    #Defining Wiener increments:
    dW = rng.normal(0, np.sqrt(timeStep), totalSteps)

    #Stationary initial condition
    ou0 = rng.normal(0, np.sqrt(1/(2*gamma)))

    #Generating the realization of the OU-process
    for i in range(totalSteps):
        ou[i] = ou0 * np.exp(-gamma*t[i]) + np.exp(-gamma*t[i]) * np.sum(np.exp(gamma*t[0:i]) * dW[0:i])


    return ou


def OU_sheet_realization(totalTime, tStep, totalLength, xStep, tgamma, xgamma):
    '''
    Function creating a realization of an OU-sheet. Generating a time realization for each step of x, 
    using a certain solution to the 2D OU-process. 
    '''


    #Initializing arrays and necessary parameters
    totaltSteps = int(totalTime/timeStep + 1) #include t=0
    t = np.arange(0, totalTime + timeStep, tStep)
    
    totalxSteps = int(totalLength/xStep + 1)
    x = np.arange(0, totalLength + xStep, xStep)

    sheet = np.empty([totalxSteps, totaltSteps])
    tprocess = np.empty([totalxSteps, totaltSteps])

    for i in range(totalxSteps):
        tprocess[i, :] = OU_time_realization(totalTime, tStep, tgamma)


    for i in range(totalxSteps):
        for j in range(totaltSteps):
            sheet[i,j] = 1/totalxSteps * np.exp(-xgamma * x[i]) * np.sum(tprocess[0:i,j] * np.exp(xgamma * x[0:i]))
            #sheet[i,j] = x0 * np.exp(-xgamma * x[i]) + xStep * np.exp(-xgamma* x[i]) * np.sum(tsheet[0:i+1, j] * np.exp(xgamma * x[0:i+1]))


    return sheet

totalTime = 5
timeStep = 0.05
t = np.arange(0, totalTime+timeStep, timeStep)

totalLength = 5
xStep = 0.05
x = np.arange(0, totalLength + xStep, xStep)


l_c = 10
xgamma = 1/l_c

t_c = 10
tgamma = 1/t_c



T, X = np.meshgrid(t, x)


sheet = OU_sheet_realization(totalTime, timeStep, totalLength, xStep, tgamma, xgamma)


def plot3D():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T, X, sheet, cmap='plasma')
    ax.set_title('2-Dimensional OU-process')
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.show()



def plot2D():
    ax = plt.axes()
    ax.imshow(sheet)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.show()



def plotSlices(slice1, slice2):
        
        plt.subplot(2, 2, 1)
        plt.plot(t, sheet[slice1, :])
        plt.title("Slice of the OU-process: xstep = " + str(slice1))
        plt.xlabel("t")
        plt.ylabel("")
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(x, sheet[:, slice1])
        plt.title("Slice of the OU-process: tstep = " + str(slice1))
        plt.xlabel("x")
        plt.ylabel("")
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(t, sheet[slice2, :])
        plt.title("Slice of the OU-process: xstep = " + str(slice2))
        plt.xlabel("t")
        plt.ylabel("")
        plt.grid()


        plt.subplot(2, 2, 4)
        plt.plot(x, sheet[:, slice2])
        plt.title("Slice of the OU-process: tstep = " + str(slice2))
        plt.xlabel("x")
        plt.ylabel("")
        plt.grid()

        plt.tight_layout(pad=0.5)
        plt.show()

def testingSheet(totalTime, timeStep, velocity, tgamma, xgamma, N, s1, s2, s3):
     
     #Sampling values at time = t for N realizations of the Ornstein-Uhlenbeck process.
    samples1 = np.empty(N)
    samples2 = np.empty(N)
    samples3 = np.empty(N)

    a1, a2 = s1
    b1, b2 = s2
    c1, c2 = s3
    

    for i in trange(N):
        process = OU_sheet_realization(totalTime, timeStep, velocity, tgamma, xgamma)
        samples1[i] = process[a1, a2]
        samples2[i] = process[b1, b2]
        samples3[i] = process[c1, c2]

    plt.hist(samples1, bins=70, density=True, alpha=0.7, label="step: "+str(s1))
    plt.hist(samples2, bins=70, density=True, alpha=0.5, label="step: "+str(s2))
    plt.hist(samples2, bins=70, density=True, alpha=0.3, label="step: "+str(s3))
    plt.grid()
    plt.legend()
    plt.show()

    return

plot3D()
plotSlices(0, 50)

#testingSheet(totalTime, timeStep, velocity, tgamma, xgamma, 100, [0,0], [0,50], [0,75])