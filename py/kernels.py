import numpy as np
import collections.abc


"""
    kernels in original and fourier tranformation - equations determined by WolframAlpha.
    originally according to publication 'Graph neural fields: 
    A framework for spatiotemporal dynamical models on the human connectome' 
    by Aqil, Atasoy, Kringelbach & Hindriks (2021), Table 2., but that was 'in the fourier domain' and not correct"""

# # # - - - kernels in original domain - - - # # #

def exponential(sigma, x):
    return (1/(2*sigma))*np.exp(-np.abs(x)/sigma)

def gaussian(sigma, x):
    return (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(((x**2))/(2*(sigma**2))))  #(1/(2*(sigma**2)))*np.exp(-((x**2)/(2*(sigma**2))))  #1/(np.pi*(sigma**2))*np.exp(-0.5*(x/sigma)**2)

def rectangular(sigma, x):
    h = x / sigma
    return np.where(np.abs(h) < 1, 1 / (2*sigma), 0)


# # # - - - kernels in Fourier domain - - - # # # EXPONENTOAL REQUIRES UPDATE ! ! !

def f_exponential(sigma, k):
    return 1 / (1 + k**2 * sigma**2) #scale*(1/((alpha**2) + (k**2)))

def f_gaussian(sigma, k):
    return np.exp(-(1/2) * (sigma**2) * (k**2) )

def f_rectangular(sigma, k):
    denominator = k * sigma
    result = np.where(np.isclose(denominator, 0), 0, np.sin(denominator) / denominator)
    result = np.where(np.isnan(result), 0, result)  # Handle NaN values from previous step
    return result



# # # - - - derivatives of kernels - - - # # #

def deriv_gaussian(sigma, x):
    return -(x/(np.sqrt(2*np.pi)*sigma**3))*np.exp(-(((x**2))/(2*(sigma**2))))


def deriv_exponential(sigma, x):
    return (1 / 2 * (sigma**2) * -np.sign(x)) * np.exp(-np.abs(x) / sigma)


def deriv_rectangular(sigma, x):
        return 0.0

# # # - - - derivatives of f_kernels - - - # # # UPDATE REQUIRED ! ! !

    
def deriv_f_gaussian(sigma, k):
    return -(sigma**2)*k*np.exp(-(1/2)*(sigma**2)*(k**2))


def deriv_f_exponential(sigma,k):
    return -(k*sigma**2)/(1 + k**2*sigma**2)**2