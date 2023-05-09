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
    
    #sheet = x0 * np.exp(-xgamma*x) + xStep * np.exp(-xgamma*x) * np.cumsum(tsheet * np.exp(xgamma*x), axis=0)
    #sheet[i, :] = sheet00 * np.exp(-xgamma * t) * np.exp(-tgamma * t) + xStep * np.exp(-xgamma * x) * np.cumsum(tprocess * np.exp(xgamma * x), axis=0)

    
    for j in range(totaltSteps):
        sheet[:, j] = xStep * np.exp(-xgamma * x) * np.cumsum(tsheet[:, j] * np.exp(xgamma * x))
        #sheet[:, j] = sheet00 * np.exp(-xgamma * x) * np.exp(-tgamma * t[j]) + xStep * np.exp(-xgamma * x) * np.cumsum(tsheet[:, j] * np.exp(xgamma * x))
        
        #sheet[:, j] = xStep * np.exp(-xgamma * x) * np.cumsum(tsheet[:, j] * np.exp(xgamma * x))

            #sheet[i,j] = sheet00 * np.exp(-xgamma * x[i]) * np.exp(-tgamma * t[j]) + xStep * np.exp(-xgamma* x[i]) * np.sum(tsheet[0:i+1, j] * np.exp(xgamma * x[0:i+1])) #pretty decent
            
            #sheet[i,j] = sheet00 * np.exp(-xgamma * x[i]) + xStep * np.exp(-xgamma* x[i]) * np.sum(tsheet[0:i+1, j] * np.exp(xgamma * x[0:i+1])) #goat
            
            #sheet[i,j] = sheet00 * np.exp(-xgamma * x[i]) * np.exp(-tgamma*t[j]) + xStep * np.exp(-xgamma* x[i]) * np.sum(tsheet[0:i+1, j] * np.exp(xgamma * x[0:i+1])) #very good yes

            
    return sheet



totalTime = 10
timeStep = 0.01
t = np.arange(0, totalTime+timeStep, timeStep)

totalLength = 10
xStep = 0.01
x = np.arange(0, totalLength + xStep, xStep)


l_c = 1
xgamma = 1/l_c

t_c = 1
tgamma = 1/t_c

N = 5000

T, X = np.meshgrid(t, x)


sheet = OU_sheet_realization(totalTime, timeStep, totalLength, xStep, tgamma, xgamma)


def plot3D():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T, X, sheet, cmap='plasma')
    ax.set_title('2-Dimensional OU-process')
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
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




def testingSheet(totalTime, timeStep, totalLength, xStep, tgamma, xgamma, N, s1, s2, s3):
     
     #Sampling values at time = t for N realizations of the Ornstein-Uhlenbeck process.
    samples1 = np.empty(N)
    samples2 = np.empty(N)
    samples3 = np.empty(N)

    a1, a2 = s1
    b1, b2 = s2
    c1, c2 = s3
    

    for i in trange(N):
        process = OU_sheet_realization(totalTime, timeStep, totalLength, xStep, tgamma, xgamma)
        samples1[i] = process[a1, a2]
        samples2[i] = process[b1, b2]
        samples3[i] = process[c1, c2]

    plt.hist(samples1, bins=50, density=True, alpha=0.6, label="step: "+str(s1))
    plt.hist(samples2, bins=50, density=True, alpha=0.4, label="step: "+str(s2))
    plt.hist(samples2, bins=50, density=True, alpha=0.2, label="step: "+str(s3))
    plt.grid()
    plt.legend()
    plt.show()

    return



def testingCorrelation(totalTime, timeStep, totalLength, xStep, tgamma, xgamma, N):
    '''
    Finds the autocorrelation function of the OU-process by averaging N samples.
    '''

    #Initializing arrays and necessary parameters
    totaltSteps = int(totalTime/timeStep + 1) #include t=0
    t = np.arange(0, totalTime + timeStep, timeStep)

    totalxSteps = int(totalLength/xStep + 1) #add x=0
    x = np.arange(0, totalLength + xStep, xStep)

    T, X = np.meshgrid(t, x)

    corrSample = np.empty([totalxSteps, totaltSteps])
    corr = np.empty([totalxSteps, totaltSteps])

    for i in trange(N):
        sheet = OU_sheet_realization(totalTime, timeStep, totalLength, xStep, tgamma, xgamma)
        
        corrSample = np.dot(sheet[0,0], sheet)
        #for j in range(totalxSteps):
            #for k in range(totaltSteps):
                #corrSample[j,k] = sheet[0,0] * sheet[j,k]
        
        corr += corrSample/N

    trueProcess = 1/(4*tgamma*xgamma) * np.exp(-tgamma * T) * np.exp(-xgamma * X)
    
    plt.subplot(2,2,1)
    plt.imshow(trueProcess)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.grid()

    plt.subplot(2,2,2)
    plt.imshow(corr)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.grid()

    plt.show()

    return



plot3D()
plotSlices(0, 50)

#testingSheet(totalTime, timeStep, totalLength, xStep, tgamma, xgamma, N, [50,10], [50,50], [50,75])
#testingCorrelation(totalTime, timeStep, totalLength, xStep, tgamma, xgamma, N)