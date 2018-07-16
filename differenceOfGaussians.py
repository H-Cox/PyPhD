import numpy as np
import gaussians as gs

def importParameters(filename):
    
	return np.genfromtxt(filename+'_param.csv',delimiter = ',')

def findDifference(parameters1, parameters2):


	xtest = np.linspace(-10,10,1000)

	y1 = gs.multiGaussianW(xtest,p1[:,0],p1[:,1],p1[:,2])
	y2 = gs.multiGaussianW(xtest,p2[:,0],p2[:,1],p2[:,2])

	ydiff = y2-y1

	ydiffok = ydiff[xtest>1]

	diff = sum(ydiffok[ydiffok > 0])

	percentageDiff = diff/np.mean([sum(y1),sum(y2)])

	return percentageDiff

if __name__ == '__main__':
	filepath = '/Users/Henry/OneDrive/PhD-Folder/GMM data/'

	sample1 = 'Sim1'
	sample2 = '8mM'
	p1 = importParameters(filepath+sample1)
	p2 = importParameters(filepath+sample2)

	diff = findDifference(p1, p2)

	print('considering two samples...')

	print('...')

	print('The difference is..')

	print(diff*100)