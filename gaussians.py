import numpy as np

def gaussian(x,mu,sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.))) / np.power(2*3.141*np.power(sigma, 2.),0.5)

def multiGaussian(x,mu,sigma):
    result = np.zeros(x.shape)
    for g in range(len(mu)):
        result = np.add(result,gaussian(x,mu[g],sigma[g]))
    return result
def multiGaussianW(x,mu,sigma,w):
    result = np.zeros(x.shape)
    for g in range(len(mu)):
        result = np.add(result,w[g]*gaussian(x,mu[g],sigma[g]))
    return result