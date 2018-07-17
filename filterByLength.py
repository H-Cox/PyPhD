import numpy as np
import gaussianMixtures as gM

def importData(filename):

    data = np.genfromtxt(filename+'.csv',delimiter = ',')

    return data[:,np.newaxis]

def filterSecond(minVal,maxVal,valArray,filterArray):
    
    return list(item for ind,item in enumerate(filterArray) if valArray[ind]>minVal and valArray[ind]<maxVal)

def minMaxList(numberofBins):
    minVal = np.log(2)
    maxVal = np.log(50)
    logList = np.linspace(minVal,maxVal,numberofBins+1)
    return np.exp(logList)

def binEnergy(maxBins,lengthData,bendData):
    binnedEnergy = []

    for nb in range(maxBins):
    
        thisList = minMaxList(nb+1)

        thisbin = []
        for item in range(len(thisList)-1):
            thisbin.append(filterSecond(thisList[item],thisList[item+1],lengthData,bendData))
        
        binnedEnergy.append(thisbin)
    
    return binnedEnergy