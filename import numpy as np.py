import numpy as np
import sklearn.mixture as mix
import pandas as pd

def importData(filename):

	data = np.genfromtxt(filename+'.csv',delimiter = ',')

	return data[:,np.newaxis]

def saveData(filename,data,means,sigmas,w,xtest,ytest,numberOfGaussians):

	i = numberOfGaussians

	means2 = np.array(means[i]).flatten()
	sigs2 = np.array(sigs[i]).flatten()
	w2 = np.array(w[i]).flatten()
	pdata = np.transpose([means2,sigs2,w2])

	counts, bins = np.histogram(data,30,density=True)

	c = counts.flatten()
	b = bins[:-1].flatten()
	b = np.add(b,(b[1]-b[0])/2)
	hdata = np.transpose([b,c])

	c = xtest.flatten()
	b = ytest[:,i]
	d = b.flatten()

	fdata =np.transpose([c,b])

	np.savetxt(filename + '_fit.csv',fdata,delimiter=',')
	np.savetxt(filename + '_param.csv',sdata,delimiter=',')
	np.savetxt(filename + '_hist.csv',hdata,delimiter=',')


def main():

	filename = '/Users/Henry/OneDrive/PhD-Folder/GMM data/Sim1t.csv'

	data = importData(filename)

	data = np.log(data)

	maxGaussians = 10

	xtest = np.linspace(np.amin(data)-1,np.amax(data)+1,1000)
	xtest = xtest[:,np.newaxis]

	# 1000 rows and max gaussians columns
	ytest = np.zeros(xtest.shape[0],maxGaussians)

	means = []
	sigs = []
	w = []

	for g in range(maxGaussians):

		GMM = mix.GaussianMixture(n_components = g+1)

		GMM = GMM.fit(data)

		means.append(GMM.means_)
		sigs.append(np.power(GMM.covariances_,0.5))
		w.append(GMM.weights_)

		ytest[:,g] = np.exp(GMM.score_samples(xtest))


	saveData(filename,data,means,sigmas,w,xtest,ytest,3)


if __name__ == '__name__':

	main()
