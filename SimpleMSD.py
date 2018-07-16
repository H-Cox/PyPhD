
def simpleMSD1D(x):
    import numpy as np
    x = np.array(x)
    y = []
    for t in range(len(x)-1):
        x1 = x[1+t:]
        x2 = x[:-1-t]
        xdiff = x1-x2
        y.append(np.sum(np.square(xdiff)))
    return y

def simpleMSD(x):
    import numpy as np
    # convert to numpy array
    x = np.array(x)
    
    # if it is one dimensional data, use the 1D code
    if x.ndim == 1:
        return simpleMSD1D(x)
    
    y = []
    # r = number of time points in the data, d = number of dimensions
    [r,d] = np.shape(x)
    
    # loop through each lag time
    for t in range(r-1):
        x1 = x[1+t:,:]
        x2 = x[:-1-t,:]
        # calculate the differences in each dimension for this lag time
        xdiff = x1-x2
        
        # average the sum of squared differences to find the MSD for this lag time
        y.append(np.mean(np.sum(np.square(xdiff),1)))
    return y