import numpy as np

# A library of many useful functions arranged in alphabetical order.

# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# 09/08/2018
# averages y values over x values which are the same.
def averageOverX(x,y):
    
    uniqueX = np.unique(x)
    
    uniqueY = np.zeros(len(uniqueX))
    
    if y.ndim==1:
        y = y[:,np.newaxis]
    
    for u,uX in enumerate(uniqueX):
        uniqueY[u] = np.nanmean(y[x==uX])
    
    return uniqueX, uniqueY

# BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
# 20/07/18
# generates an array of samples using bootstrap sampling. This can then be run iteratively
# on a function to generate bootstrapped data for error analysis etc...
def bootstrap(data,n_samples,returnFlat = False,returnIndices = False):
    
    data = np.array(data)
    
    originalShape = data.shape
    
    flatData = data.flatten()
    
    dataSize = len(flatData)
    
    bootstrapSamples = np.zeros((dataSize,n_samples))
    indices = np.zeros(bootstrapSamples.shape,dtype='int')
    
    for s in range(n_samples):
        
        indices[:,s] = np.random.choice(list(range(dataSize)),dataSize)
        
        bootstrapSamples[:,s] = flatData[indices[:,s],]
    
    if not returnFlat:
        output = []
        for s in range(n_samples):
            output.append(bootstrapSamples[:,s].reshape(originalShape))
    else:
        output = bootstrapSamples
    
    if output[0].ndim == 1:
        output = np.vstack(output).T
    
    if returnIndices:
        return output, indices
    else:
        return output

# 09/08/2018
# does histcounts but using a bootstrap so that error bars are also generated.
def bootstrapHist(data,minimum=None,maximum=None,numberOfBins=50,logspaceYN=False,nBootstraps=1000):
    
    if minimum == None:
        minimum = min(data)
    if maximum == None:
        maximum = max(data)
    
    edges, x = makeBins(logspaceYN=logspaceYN,maximum=maximum,minimum=minimum,numberOfBins=numberOfBins)
    
    bootdata = bootstrap(data=data,n_samples=nBootstraps)
    
    yData = histcounts(edges,bootdata)
    
    y = np.vstack([np.average(yData,axis=1),np.std(yData,axis=1)]).T
        
    return x, y    
    
# CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
# 26/07/18
def chiSquared(yexp,yfit,yerror,degreesOfFreedom=1):
    
    variance= np.power(yerror,2)
    
    chi2 = np.sum(np.divide(np.power(yexp-yfit,2),variance))
    
    return chi2/degreesOfFreedom

# EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
# 09/08/2018
# calculates the empirical cumulative distribution function
def ecdf(data,uniqueData = True):
    
    sortedData = np.sort(data,axis=0)
    
    x = sortedData
    y = np.linspace(1/len(x),1,len(x))
    
    if uniqueData:
        x,y = averageOverX(x,y)
    
    return x,y

# FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
# 17/07/18
# finds the minimum and maximum value of a list of lists (or np arrays)
def findMinMax(listOfLists):
    
    fullList = fullFlatten(listOfLists)
    
    return np.amax(fullList), np.amin(fullList)
# 17/07/18
# flattens a list of lists (or np arrays) into one long list
def fullFlatten(listOfLists):
    
    flatList = []
    
    for subList in listOfLists:
        flatSubList = np.array(subList).flatten()
        for item in  flatSubList:
            flatList.append(item)
    
    return flatList

# GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
# 18/07/18
# generates a grid of points arranged in a 2-column vector, used for sampling a 2D surface
def generateGrid(maxX, maxY, minX = 0, minY = 0, n_points = 1000):
    x = np.linspace(minX,maxX,n_points)
    y = np.linspace(minY,maxY,n_points)
    xx,yy = np.meshgrid(x,y)
    X = xx.flatten()
    Y = yy.flatten()
    XY = np.vstack((X,Y)).T
    return XY, x, y

# HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# 19/07/18
# turns any value of zero into a nan
def HenryMethod(data):
    
    data = np.array(data,dtype=float)
    
    data[data==0] = np.nan
    
    return data
# 20/07/18
# returns the histogram bin values for a specified range of bin edges,
# works with a list of multiple sets of data.
def histcounts(edges,inputs,density=True):
    
    inputs = np.array(inputs)
    
    if inputs.ndim == 1:
        n_samples = inputs.shape[0]
        array2d = False
    else:
        n_samples = inputs.shape[1]
        array2d = True
        
    counts = np.zeros((len(edges)-1,n_samples))
    
    for c in range(n_samples):
        if array2d:
            counts[:,c], bins = np.histogram(inputs[:,c],edges,density=density)
        else:
            counts[:,c], bins = np.histogram(inputs[c],edges,density=density)
    
    return np.vstack(counts)

# MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# 19/07/18
# generates a list of bin edges and a list of the centre of each bin for histogramming
def makeBins(minimum,maximum,numberOfBins=10,logspaceYN=False):
    
    if logspaceYN:
        
        binEdges = np.logspace(np.log10(minimum),np.log10(maximum),numberOfBins+1)
        
        logbinEdges = np.log(binEdges)
        
        xbins = np.exp(logbinEdges[:-1]+(logbinEdges[1]-logbinEdges[0])/2)
        
    else:
        
        binEdges = np.linspace(minimum,maximum,numberOfBins+1)
        
        xbins = binEdges[:-1]+(binEdges[1]-binEdges[0])/2
        
    return binEdges, xbins

# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# 09/08/2018
# given a list of x values and a array with a first column of x and corresponding y values
# this picks the x and corresponding y values from the array which match each x in the list.
def pickSamples(xSample, xyData):
    
    xySample = np.zeros((len(xSample),xyData.shape[1]))
    
    for ind, sample in enumerate(xSample):
        
        xDiff = np.abs(xyData[:,0]-sample)
        
        indices = xDiff==min(xDiff)
        insert = xyData[indices,:]
        choice = np.random.permutation(insert.shape[0])
        
        xySample[ind,:] = insert[choice[0],:]
        
    return xySample

# 09/08/2018
# finds real x values of a polynomial with coefficients specified by p at a specific value
# of y, defaults to y = 0, set limits to x as well if you want. Had to add the loop in there
# as had trouble with multiple calls of the function modifying p permanently.
def polyval(p,y=0,xmin=-np.inf,xmax=np.inf):
    if xmin != -np.inf:
        xmin = xmin - 0.1
    
    if xmax != np.inf:
        xmax = xmax + 0.1
    
    polynomials = []
    for coeff in p:
        polynomials.append(coeff)
    polynomials[-1] = polynomials[-1]-y
    r = np.roots(p=polynomials)
    s = r[np.isreal(r)].real
    s = s[s<xmax]
    s = s[s>xmin]
    return s

# RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
# 09/08/2018
# given a list of data points this function approximates the probability distribution and then
# generates new samples that match the raw datas probability distribution.
def resampleData(x,nsamples=1000):
    
    ux,uy = ecdf(x)
    
    p = np.polyfit(ux,uy,deg = 10)
    
    ysample = np.random.rand(nsamples)
    xsample = np.zeros(nsamples)
    
    for s, sample in enumerate(ysample):
        result = polyval(p=p,y=sample,xmin=0)
        xsample[s] = result[0]
        
    return xsample

# SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
# 19/07/18
# removes any row in the data that contains NaNs
def scrubNaNs(data):
    return data[~np.isnan(data).any(axis=1)]

# 17/07/18
# finds the length of each element of the sublist in a list of lists
def subListLengths(listOfLists):
    
    return [len(subList) for subList in listOfLists]