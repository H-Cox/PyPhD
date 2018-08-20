import numpy as np
import matplotlib.pyplot as plt
import sklearn.mixture as mix
import generalFunctions as gF

# fit a 2D GMM to some data and optionally plot the result if a filename is supplied.
def plot2DGMM(data,logDatayn = True,logLikelyhoodyn=True,filename = False,plotyn = True,components = 2):
    GMM = mix.GaussianMixture(n_components = components, init_params = 'random')
    if logDatayn:
        GMM.fit(np.log(data))
    else:
        GMM.fit(data)
    
    maxX = np.log(50) #data[:,0].max()+10
    minX = np.log(2)    #data[:,0].min()
    maxY = np.log(5000) #data[:,1].max()+10
    minY = np.log(0.0001)    #data[:,1].min()
    
    XY,x,y = gF.generateGrid(maxX,maxY,minX,minY)
    
    if logDatayn:
        Z = GMM.score_samples(XY)
    else:
        Z = GMM.score_samples(np.exp(XY))
    
    if not logLikelyhoodyn:
        Z = np.exp(Z)
    
    z = Z.reshape((1000,1000))
    
    if plotyn:
        plotPointsWContour(x=x,y=y,z=z,points=data,filename=filename)
    
    return GMM

# plot a contour graph with raw data points to show fitting of a 2D GMM to data
def plotPointsWContour(x=[],y=[],z=[],points=[],filename=False, filepath = 'C:/Users/mbcx9hc4/Dropbox (The University of Manchester)/Data/Processing/18-07/'):
    if len(points)!=0:
        plt.plot(points[:,0],points[:,1],'.')
    if len(x)!=0 and len(y)!=0 and len(z)!=0:
        plt.contourf(np.exp(x),np.exp(y),z,100)
        
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(2, 50)
    plt.ylim(0.0001, 5000)
   
    plt.ylabel('Bend Energy')
    plt.xlabel('Contour length')
    plt.title('Log likelyhood of each point')
    plt.colorbar()
    if filename:
        plt.savefig(filepath + filename+ '.png',dpi=1000)