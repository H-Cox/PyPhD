import numpy as np
import matplotlib.pyplot as plt
import testingSPS as tS
import generalFunctions as gF
import scipy.stats as sS
import mathFunctions as mF
import scipy.optimize as sO

filepath = 'C:/Users/mbcx9hc4/Dropbox (The University of Manchester)/Data/Processing/18-07/'
def bendEnergyDistribution(bendEnergy,beta,U0):
    const = 1/(np.power(U0,1/beta)*beta)
    denominator = np.power(bendEnergy,(beta-1)/beta)
    numerator = np.exp(-np.power(bendEnergy/U0,1/beta))
    return const*np.divide(numerator,denominator)

def logBendEnergyDistribution(bendEnergy,beta,U0):
    return np.log(bendEnergyDistribution(bendEnergy,beta,U0))

def cleanData(x,y):

    xOut = x[y[:,0]>0]
    yOut = y[y[:,0]>0,:]
    
    xOut = xOut[yOut[:,0]>yOut[:,1]]
    yOut = yOut[yOut[:,0]>yOut[:,1],:]
    
    return xOut, yOut

def findEnergy(energy,fit):
    out = sA.fitParts(energy,fit)
    pSt = out[:,1]
    pUt = out[:,2]
    return np.abs(1-np.divide(pSt,pUt))

def fitParts(x,fit):
    
    output = np.zeros((len(x),4))
    
    output[:,0] = x
    output[:,3] = fit10mM(x,*fit)
    output[:,2] = fit[0]*fit10mM(x,*[1,*fit[1:]])
    output[:,1] = (1-fit[0])*sS.levy.pdf(x=x,loc=fit[1],scale=fit[2])
    
    return output

def fit10mM(x,weight,loc,scale):
    #old beta and U0
    beta,U0 = [1.47146671, 1.23149996]
    
    #new beta and U0 from 27
    #beta,U0 = [1.4573319,  1.76204006]
    
    # from 30a
    #beta, U0 = [1.48050821, 1.78312103]
     #from 30c
    #beta, U0 = [1.42810467, 1.76092011]
    x1 = weight*bendEnergyDistribution(bendEnergy=x,U0=U0,beta=beta)
    x2 = (1-weight)*sS.levy.pdf(x=x,loc=loc,scale=scale)
    return x1+x2

def fit10mM2(x,weight,loc,scale,beta,U0):

    x1 = weight*bendEnergyDistribution(bendEnergy=x,U0=U0,beta=beta)
    x2 = (1-weight)*sS.levy.pdf(x=x,loc=loc,scale=scale)
    return x1+x2

def fit10mML(x,weight,loc,scale):
    return np.log(fit10mM(x,weight,loc,scale))

def fit10mML2(x,weight,loc,scale,beta,U0):
    return np.log(fit10mM2(x,weight,loc,scale,beta,U0))

def fitToFindSPS(experimentalFibrils,simulatedFibrils,nbins = 50, plotYN = True,name = 'most recent fit',filepath = 'C:/Users/mbcx9hc4/Dropbox (The University of Manchester)/Data/Processing/18-08/',epsYN=False):
    
    eF = experimentalFibrils
    sF = simulatedFibrils
    
    Lsample = gF.resampleData(x=eF[:,0])
    LUsample = gF.pickSamples(xSample=Lsample,xyData=sF)
    
    xs,ys = gF.bootstrapHist(LUsample[:,1],0.0001,10000,nbins,True)
    xs,ys = cleanData(xs,ys)
    
    xe,ye = gF.bootstrapHist(eF[:,1],0.0001,10000,nbins,True)
    xe,ye = cleanData(xe,ye)
    
    fitS, qcovs = sO.curve_fit(logBendEnergyDistribution,xs,np.log(ys[:,0]),sigma=np.divide(ys[:,1],ys[:,0]),bounds=[[1,0],[2,10]])
    
    beta = [fitS[0]-np.sqrt(qcovs[0,0]),fitS[0]+np.sqrt(qcovs[0,0])]
    U0 = [fitS[1]-np.sqrt(qcovs[1,1]),fitS[1]+np.sqrt(qcovs[1,1])]
    
    fit, qcov = sO.curve_fit(fit10mML2,xe,np.log(ye[:,0]),sigma=np.divide(ye[:,1],ye[:,0]),bounds=[[0,0,0,beta[0],U0[0]],[1,np.inf,np.inf,beta[1],U0[1]]],p0=[0.8,9.5,13,fitS[0],fitS[1]])
    
    result = np.vstack((fit,np.sqrt(np.diag(qcov)))).T
    
    if plotYN:
        x = x = np.logspace(-4,4,1000)
        F = fit10mM2(x,*fit)
        plt.errorbar(x=xe,y=ye[:,0],yerr=ye[:,1],zorder = 0)
        plt.plot(x,F,zorder = 10)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(0.000001,100)
        if epsYN:
            plt.savefig(filepath+name,format = 'eps')
        else:
            plt.savefig(filepath+name,dpi = 1000)
    
    return result

def importData(filename,bins=30):
    
    bendEnergy = np.genfromtxt(filename+'.csv',delimiter = ',')

    bendEnergy = bendEnergy[:,np.newaxis]
        
    edges, x = gF.makeBins(logspaceYN=True,maximum=5000,minimum=0.0001,numberOfBins=bins)
    
    bootstrap = gF.bootstrap(data=bendEnergy,n_samples=1000)
    
    yData = gF.histcounts(edges,bootstrap)
    
    y = np.vstack([np.average(yData,axis=1),np.std(yData,axis=1)]).T
        
    return x, y
