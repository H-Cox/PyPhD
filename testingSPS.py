import numpy as np
import sklearn.mixture as mix
import generalFunctions as gF
import gaussianMixtures as gM
import filterByLength as fL

def importData(filename):

    data = np.genfromtxt(filename+'.csv',delimiter = ',')

    return data[:,np.newaxis]

# 17/07/18
# script to import data and present it as a 2D array.
def importAndPrepData(filename,filepath="C:/Users/mbcx9hc4/OneDrive/PhD-Folder/GMM data/"):
    
    bend = importData(filepath+filename)
    length = importData(filepath+'len'+filename)
    
    return np.hstack((length,bend))

# use 2D arrays for the information, data = [length, energy]
def SPSBootstrapAnalysis(refData,testData,n_bootstraps):
    
    samples = testData.shape[0]
    
    # bootsample = np.zeros((samples,n_bootstraps))
    
    refBinnedBend = fL.binEnergy(maxBins=4,bendData=refData[:,1],lengthData=refData[:,0])
    
    meanSPS = []
    GMMSPS = []
    for n in range(n_bootstraps):
        print('Bootstrap number {} of {}'.format(n+1,n_bootstraps))
        ind = np.random.choice(list(range(samples)),samples)
        
        bootSample = testData[ind,]
        
        testBinnedBend = fL.binEnergy(maxBins=4,bendData=bootSample[:,1],lengthData=bootSample[:,0])
        
        nbins, tempMeanSPS, tempGMMSPS = findSPSFromBinnedBend(refBinnedBend, testBinnedBend)
        
        meanSPS.append(tempMeanSPS)
        GMMSPS.append(tempGMMSPS)
    
    return meanSPS, GMMSPS


# Performs import, binning and SPS analysis of a test file using the reference file.
def SPSAnalysis(refFile,testFile,filepath="C:/Users/mbcx9hc4/OneDrive/PhD-Folder/GMM data/"):

    refbend = gM.importData(filepath+refFile)
    testbend = gM.importData(filepath+testFile)

    reflen = gM.importData(filepath+'len'+refFile)
    testlen = gM.importData(filepath+'len' +testFile)
    
    refBinnedBend = fL.binEnergy(maxBins=10,bendData=refbend,lengthData=reflen)
    testBinnedBend = fL.binEnergy(maxBins=10,bendData=testbend,lengthData=testlen)
    
    nbins, meanSPS, GMMSPS = findSPSFromBinnedBend(refBinnedBend, testBinnedBend)
    
    return nbins, meanSPS, GMMSPS

# Using binned sets of data the function calculated the proportion of the test data 
# which does not match that of the reference data.
def findSPSFromBinnedBend(refBinnedBend, testBinnedBend,mean_threshold = 3):

    meanSPS = []
    GMMSPS = []

    for b,bn in enumerate(refBinnedBend):
        tempMeanSPS = []
        tempGMMSPS = []
        for sub,subn in enumerate(bn):
        
            thisRefBin = refBinnedBend[b][sub]
            thisTestBin = testBinnedBend[b][sub]
        
            tempMeanSPS.append(findSPSMean(refData=thisRefBin,testData=thisTestBin,threshold=mean_threshold))
            tempGMMSPS.append(findSPSGMM(refData=np.log(thisRefBin),testData=np.log(thisTestBin)))
        
        binLengths = np.array(gF.subListLengths(testBinnedBend[b]))
        
        weights = binLengths/np.sum(binLengths) 
    
        tempMeanSPS = np.array(tempMeanSPS)[:,0]
        tempGMMSPS = np.array(tempGMMSPS)[:,0]
        
        meanSPS.append(np.average(a=tempMeanSPS,weights=weights))
        GMMSPS.append(np.average(a=tempGMMSPS,weights=weights))
        
        nbins = gF.subListLengths(refBinnedBend)
        
    return nbins, meanSPS, GMMSPS

# 16/07/18
# compares a reference data set to test data and determines the states of pre-stress 
# using the GMM method for each set of test data.
def findSPSGMM(refData,testData):
    
    testData = np.array(testData)
    refData = np.array(refData)
    
    if testData.ndim == 1:
        testData = testData[:,np.newaxis]
    
    if refData.ndim == 1:
        refData = refData[:,np.newaxis]
    
    reflen = len(refData)
    testlen = testData.shape[0]
    
    ref_components = 20 if reflen >= 20 else reflen
    test_components = 20 if testlen >= 20 else testlen
    
    inputMax, inputMin = gF.findMinMax([refData,testData])
    
    xtest = np.linspace(inputMin,inputMax+1,1000)
    xtest = np.array(xtest[:,np.newaxis])

    ytest = np.zeros((xtest.shape[0],testData.shape[1]+1))
    
    refGMM = mix.GaussianMixture(n_components = ref_components, init_params = 'random',n_init=10)
    refGMM = refGMM.fit(refData)
    ytest[:,0] = np.exp(refGMM.score_samples(xtest))
    
    for t in range(testData.shape[1]):
        testGMM = mix.GaussianMixture(n_components = test_components, init_params = 'random',n_init=10)
        testGMM = testGMM.fit(testData[:,t:t+1])
    
    
        ytest[:,t+1] = np.exp(testGMM.score_samples(xtest))
    
    ydiff = ytest[:,1:]-ytest[:,0][:,np.newaxis]
    
    cSSS = 100*np.sum(np.abs(ydiff),axis=0)/np.mean(np.sum(ytest,axis=0))/2
    
    return cSSS

# same as findSPSGMM but uses the mean and standard deviation to define a cut off energy
def findSPSMean(refData,testData,threshold):
    
    testData = np.array(testData)
    refData = np.array(refData)
    
    refMean = np.mean(refData)
    
    refStd = np.std(refData)
    
    cutoff = refMean+threshold*refStd
    
    cSSS = []
    SPSData = []
    
    if testData.ndim > 1:
        cols = testData.shape[1]
    else:
        cols = 1
        testData = testData[:,np.newaxis]
    
    for t in range(cols):
        
        SPSData.append([item for item in testData[:,t] if item>cutoff])
    
        cSSS.append(100*len(SPSData[t])/len(testData))
    
    return cSSS


def SPSTester(data, pSSS, stressFactor):
    
    nSSS = pSSS*len(data)//100
    
    sD = np.zeros((len(data),len(pSSS)+1))
    
    sD[:,0]=data[:,0]
    
    for q in range(len(nSSS)):
        
        rdata = np.random.permutation(data)
        sD[:,q+1] = np.append(rdata[:nSSS[q]]+stressFactor, rdata[nSSS[q]:])
    
    xtest = np.linspace(np.amin(sD),np.amax(sD)+1,1000)
    xtest = xtest[:,np.newaxis]
    ytest = np.zeros((xtest.shape[0],len(pSSS)+1))
    
    for q in range(len(nSSS)+1):
        
        GMM = mix.GaussianMixture(n_components = 20, init_params = 'random',n_init=10)
    
        GMM = GMM.fit(sD[:,q][:,np.newaxis])
    
        ytest[:,q] = np.exp(GMM.score_samples(xtest))
    
    ydiff = ytest[:,1:]-ytest[:,0][:,np.newaxis]
    
    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1) # two axes on figure
    ax1.plot(xtest, ytest)
    ax2.plot(xtest, ydiff)
    
    cSSS = 100*np.sum(np.abs(ydiff),axis=0)/np.mean(np.sum(ytest,axis=0))/2
    
    return xtest, ytest, ydiff,cSSS ,pSSS

def makeStressedFibrils(data,nSSS):
    
    sD = np.zeros((len(data),len(nSSS)+1))
    
    sD[:,0]=data[:,0]
    
    for q in range(len(nSSS)):
        
        rdata = np.random.permutation(data)
        sD[:,q+1] = np.append(rdata[:nSSS[q]]+1, rdata[nSSS[q]:])
    
    return sD

def SPSTestRunner(data):

    pSSS = np.array(range(0,100,10))
    
    runs = 10
    
    cSSS = np.zeros((len(pSSS),runs))
    
    for r in range(runs):
        
        x,y,yd,cSSS[:,r],pt = SPSTester(data,pSSS,1)
        
    fSSS = np.mean(cSSS,axis=1)
    
    return pSSS, fSSS
