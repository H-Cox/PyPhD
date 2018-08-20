import numpy as np

def exponential(x,coefficients):
    return coefficients[0]*np.exp(-x/coefficients[1])

def linear(x,m,c):
    return m*x+c

def powerLaw(x,alpha,constant):
    return constant*np.power(x,alpha)
