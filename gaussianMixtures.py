import numpy as np
import sklearn.mixture as mix
import pandas as pd


def importData(filename):

	data = np.genfromtxt(filename+'.csv',delimiter = ',')

	return data[:,np.newaxis]

def packageData(data,means,sigs,w,xtest,ytest,numberOfGaussians):

	i = numberOfGaussians
	counts, bins = np.histogram(data,30,density=True)

	means2 = np.array(means[i]).flatten()
	sigs2 = np.array(sigs[i]).flatten()
	w2 = np.array(w[i]).flatten()
	pdata = np.transpose([means2,sigs2,w2])

	c = counts.flatten()
	b = bins[:-1].flatten()
	b = np.add(b,(b[1]-b[0])/2)
	hdata = np.transpose([b,c])

	c = xtest.flatten()
	b = ytest[:,i]
	d = b.flatten()

	fdata =np.transpose([c,b])

	return [pdata, hdata, fdata]

def saveData(filename,data,means,sigs,w,xtest,ytest,numberOfGaussians):

	print('saving data')

	[pdata, hdata, fdata] = packageData(data,means,sigs,w,xtest,ytest,numberOfGaussians)
	
	np.savetxt(filename + '_fit.csv',fdata,delimiter=',')
	np.savetxt(filename + '_param.csv',pdata,delimiter=',')
	np.savetxt(filename + '_hist.csv',hdata,delimiter=',')
	
	print('done')

def main(filename):

	data = importData(filename)

	data = np.log(data)
	print('imported file')
	maxGaussians = 10

	xtest = np.linspace(np.amin(data)-1,np.amax(data)+1,1000)
	xtest = xtest[:,np.newaxis]

	# 1000 rows and max gaussians columns
	ytest = np.zeros((xtest.shape[0],maxGaussians))

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

	print('done fitting')

	saveData(filename,data,means,sigs,w,xtest,ytest,3)


if __name__ == '__main__':
	filename = '/Users/Henry/OneDrive/PhD-Folder/GMM data/Sim1t'
	main(filename)
