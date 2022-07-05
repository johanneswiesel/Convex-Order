from generate_alpha import *
from ot.datasets import make_1D_gauss as gauss
import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
import scipy as scipy
from ot.datasets import make_1D_gauss as gauss
from ot.datasets import make_2D_samples_gauss as gauss2
from scipy.stats import dirichlet
import math
from scipy import stats
from itertools import product
from scipy.spatial.distance import cdist

# Authors: Authors: Johannes Wiesel, Erica Zhang
# Version: June 30, 2022

# DESCRIPTION: This package provides optimization tools to generate the minimal Wasserstein distance under an optimal measure 'rho'


def generate_signs(n):
    r"""Generate 'n' -1 or 1

    The main idea of this function is to randomly generate 'n' signs 

    Parameters
    ----------
    n : int
        number of signs generated
        
    Returns
    -------
    mySigns : list, int
        list of length 'n' where each entry is either -1 or 1
    """
    mySigns = []
    ls = [-1,1]
    for i in range(n):
        signs = np.random.choice(ls)
        mySigns.append(signs)
    return np.array(mySigns) 



def generate_an_optimal(n = 1000,p = 10, a=gauss(100, m=15, s=5),b = gauss(100, m=10, s=1),a_grid = np.arange(100, dtype=np.float64), b_grid = np.arange(100, dtype=np.float64), target_size = 100, lbd = 1, ubd = 101,  alpha_size = 10, strata_nb = 2, method = 'random'):
    r""" Return the minimal Wasserstein distance under an optimal measure 'rho' using "Histogram" method for d = 1

    The main idea of this function is to calculate the minimal Wasserstein distance under an optimal measure 'rho'. This method models a general discrete probability measure 'rho' through using a dirichlet distribution to model the probability mass across its finite support. As the partitioned grid (which represents the discretized support of 'rho') gets finer, rho approximates to having a continuous support. For dimension one, this method calculates the minimum through using the histograms of the distributions. 
    
    Parameters
    ----------
    n : int
        number of the Dirichlet parameter alpha sampled
    p : int
        number whereby we partition the grid for the support of the optimal measure 'rho'
    a : ndarray, float64
        ndarray of the first source distribution. It should be of shape(n,d), where d is the dimension. For d = 1, 'a' is the ndarray of sample weights (or histograms) of the source distribution. For d > 1, 'a' is simply the ndarray of samples 
    b : ndarray, float64
        ndarray of the second source distribution. It should be of shape(n,d), where d is the dimension. For d = 1, 'b' is the ndarray of sample weights (or histograms) of the source distribution. For d > 1, 'b' is simply the ndarray of samples
    a_grid : ndarray, float64
        ndarray of the histogram grid for the first source distribution. This parameter is only required for d = 1
    b_grid : ndarray, float64
        ndarray of the histogram grid for the second source distribution. This parameter is only required for d = 1
    target_size : int
        the number of bins (for d=1) or number samples (for d>1) requested for the optimal measure 'rho'
    lbd : float64
        the lower bound of the alpha parameter for Dirichlet distribution
    ubd : float64
        the upper bound of the alpha parameter for Dirichlet distribution
    alpha_size : int
        number of sub-parts whereby we partition the grid in each dimension of the alpha parameter
    strata_nb : int
        number of alpha entries we generate for each sub-part of the partitioned grid of the alpha parameter for each dimension
    method : str
        the method that we use to sample alpha. Legal names include: "random", "auto random strata probing", "auto random strata probing zip", "random strata probing", and "extrema strata probing"
               
    Returns
    -------
    myMin : float64
        a single float that refers to the minimal Wasserstein distance found across all sampled alpha parameters
    """
    source_a_size = len(a)
    source_b_size = len(b)
    # cast input into ndarray
    a = np.array(a)
    b = np.array(b)
    # check input validity wrt dimensions
    if a.ndim != b.ndim:
        raise ValueError("Your input source distributions should come from the same dimension!")
    d = a.ndim    
    # check validity of input
    if source_a_size != len(a_grid) or source_b_size != len(b_grid):
        raise ValueError("Your source distribution size and its corresponding grid size do not match!")
    
    # convert passed-in grids to np.array
    a_grid = np.array(a_grid)
    b_grid = np.array(b_grid)

    # if grid is one-dimensional
    if d == 1:
        # (1). set up the grid 
        rho_grid = np.linspace(-1, 1, p) # target bin positions
        # (2). calculate cost matrixes
        # target size for here?
        M2_a = cdist(a_grid.reshape((source_a_size, 1)), rho_grid.reshape((p, 1)),  lambda u, v: -np.dot(u,v))
        M2_b = cdist(b_grid.reshape((source_b_size, 1)), rho_grid.reshape((p, 1)),  lambda u, v: -np.dot(u,v))
        # (3). generate n samples of alpha, the p-dim vector
        alpha = systematic_generate_alpha(n,p,lbd,ubd,size = alpha_size, strata_nb = strata_nb, method = method)
        result = []
        for i in range(len(alpha)):
            dir_probability = dirichlet.rvs(alpha[i])
            # unravel the ndarray
            rho_rv = dir_probability.ravel()
            diff = -ot.emd2(a, rho_rv, M2_a)+ ot.emd2(b, rho_rv, M2_b)
            result.append(diff)
        mx = np.array(result).min()
        return mx 
    else:
        rho_num = p**d # total number of categorical variables
        x1 = np.ones((source_a_size,)) / source_a_size # uniform distribution on samples (array of bin heights rather than bin positions!)
        x2 = np.ones((target_size,)) / target_size
        x3 = np.ones((source_b_size,)) / source_b_size
        # generate n samples of alpha, the p-dim vector
        alpha = systematic_generate_alpha(n,rho_num,lbd,ubd,size = alpha_size, strata_nb = strata_nb, method = method)
        # d-D grid
        grid = np.array([np.linspace(-1, 1, p)])
        multi_grid = np.repeat(grid,d,axis=0)
        coordinates = np.array(np.meshgrid(*multi_grid)).T.reshape(-1, d) # ndarray of all the discrete points in the grid
        result = []
        for i in range(len(alpha)):
            dir_probability = dirichlet.rvs(alpha[i])
            # unravel the ndarray
            rho_rv = dir_probability.ravel() # discrete probability of each point of the grid
            xk = np.arange(rho_num)
            custm = stats.rv_discrete(name='custm', values=(xk, rho_rv))
            R = custm.rvs(size=target_size) # index array
            rho = coordinates[R]
            M2_a = cdist(a, rho, lambda u, v: -np.dot(u,v)) # loss matrix        
            M2_b = cdist(b, rho, lambda u, v: -np.dot(u,v)) # loss matrix        
            diff = -ot.emd2(x1, x2, M2_a)+ ot.emd2(x3,x2, M2_b)
            result.append(diff)
        myMin = np.array(result).min()
        return myMin
    

def generate_an_optimal_samples(a,b,n = 1000,p=10, target_size = 100, lbd = 1, ubd = 101, alpha_size = 10, strata_nb = 2, method = 'random'):
    r""" Return the minimal Wasserstein distance under an optimal measure 'rho' using "samples" method

    The main idea of this function is to calculate the minimal Wasserstein distance under an optimal measure 'rho'. This method models a general discrete probability measure 'rho' through using a dirichlet distribution to model the probability mass across its finite support. As the partitioned grid (which represents the discretized support of 'rho') gets finer, rho approximates to having a continuous support. This method uses distribution samples for across all dimensions. 
    
    Parameters
    ----------
    a : ndarray, float64
        ndarray of the first source distribution. It should be of shape(n,d), where d is the dimension. For all dimensions, 'a' is simply the ndarray of samples 
    b : ndarray, float64
        ndarray of the second source distribution. It should be of shape(n,d), where d is the dimension. For all dimensions, 'b is simply the ndarray of samples
    n : int
        number of the Dirichlet parameter alpha sampled
    p : int
        number whereby we partition the grid for the support of the optimal measure 'rho'
    target_size : int
        the number of bins (for d=1) or number samples (for d>1) requested for the optimal measure 'rho'
    lbd : float64
        the lower bound of the alpha parameter for Dirichlet distribution
    ubd : float64
        the upper bound of the alpha parameter for Dirichlet distribution
    alpha_size : int
        number of sub-parts whereby we partition the grid in each dimension of the alpha parameter
    strata_nb : int
        number of alpha entries we generate for each sub-part of the partitioned grid of the alpha parameter for each dimension
    method : str
        the method that we use to sample alpha. Legal names include: "random", "auto random strata probing", "auto random strata probing zip", "random strata probing", and "extrema strata probing"
               
    Returns
    -------
    myMin : float64
        a single float that refers to the minimal Wasserstein distance found across all sampled alpha parameters
    """
    source_a_size = len(a)
    source_b_size = len(b)
    x1 = np.ones((source_a_size,)) / source_a_size # uniform distribution on samples (array of bin heights rather than bin positions!)
    x2 = np.ones((target_size,)) / target_size
    x3 = np.ones((source_b_size,)) / source_b_size
    # cast input into ndarray
    a = np.array(a)
    b = np.array(b)
    # check input validity wrt dimensions
    if a.ndim != b.ndim:
        raise ValueError("Your input source distributions should come from the same dimension!")
    # read dimension
    d = a.ndim
    rho_num = p**d # total number of categorical variables
    # generate n samples of alpha, the p-dim vector
    alpha = systematic_generate_alpha(n,rho_num,lbd,ubd,size = alpha_size, strata_nb = strata_nb, method = method)
    # d-D grid
    if d == 1:
        coordinates = np.linspace(-1, 1, p)        
    else:
        grid = np.array([np.linspace(-1, 1, p)])
        multi_grid = np.repeat(grid,d,axis=0)
        coordinates = np.array(np.meshgrid(*multi_grid)).T.reshape(-1, d) # ndarray of all the discrete points in the grid
    result = []
    for i in range(len(alpha)):
        dir_probability = dirichlet.rvs(alpha[i])
        # unravel the ndarray
        rho_rv = dir_probability.ravel() # discrete probability of each point of the grid
        xk = np.arange(rho_num)
        custm = stats.rv_discrete(name='custm', values=(xk, rho_rv))
        R = custm.rvs(size=target_size) # index array
        rho = coordinates[R]
        M2_a = cdist(a.reshape(source_a_size,d), rho.reshape(target_size,d),lambda u, v: -np.dot(u,v)) # loss matrix        
        M2_b = cdist(b.reshape(source_b_size,d), rho.reshape(target_size,d),lambda u, v: -np.dot(u,v)) # loss matrix        
        diff = -ot.emd2(x1, x2, M2_a)+ ot.emd2(x3,x2, M2_b)
        result.append(diff)
    myMin = np.array(result).min()
    return myMin  

    

def generate_an_optimal_dirichlet(a,b,n = 10,target_size = 100, lbd = 1, ubd = 101, alpha_size = 10, strata_nb = 2, method = 'random'):
    r""" Return the minimal Wasserstein distance under an optimal measure 'rho' 

    The main idea of this function is to calculate the minimal Wasserstein distance under an optimal measure 'rho'. Instead of using Dirichlet distribution to model the probability mass function for a general discrete probability measure 'rho', this method samples 'rho' directly through Dirichlet random variables. We allow for negative values by randomly generating signs to the Dirichlet random variables. To sample the Dirichlet random variables, this method selects the generated alphas randomly. 
    
    Parameters
    ----------
    a : ndarray, float64
        ndarray of the first source distribution. It should be of shape(n,d), where d is the dimension. For all dimensions, 'a' is simply the ndarray of samples 
    b : ndarray, float64
        ndarray of the second source distribution. It should be of shape(n,d), where d is the dimension. For all dimensions, 'b is simply the ndarray of samples
    n : int
        number of the Dirichlet parameter alpha sampled
    target_size : int
        the number of bins (for d=1) or number samples (for d>1) requested for the optimal measure 'rho'
    lbd : float64
        the lower bound of the alpha parameter for Dirichlet distribution
    ubd : float64
        the upper bound of the alpha parameter for Dirichlet distribution
    alpha_size : int
        number of sub-parts whereby we partition the grid in each dimension of the alpha parameter
    strata_nb : int
        number of alpha entries we generate for each sub-part of the partitioned grid of the alpha parameter for each dimension
    method : str
        the method that we use to sample alpha. Legal names include: "random", "auto random strata probing", "auto random strata probing zip", "random strata probing", and "extrema strata probing"
               
    Returns
    -------
    myMin : float64
        a single float that refers to the minimal Wasserstein distance found across all sampled alpha parameters
    """
    # cast input into ndarray
    a = np.array(a)
    b = np.array(b)
    # check input validity wrt dimensions
    if a.ndim != b.ndim:
        raise ValueError("Your input source distributions should come from the same dimension!")
    d = a.ndim
    dirichlet_dim = d+1
    source_a_size = len(a)
    source_b_size = len(b)
    x1 = np.ones((source_a_size,)) / source_a_size # uniform distribution on samples
    x2 = np.ones((target_size,)) / target_size
    x3 = np.ones((source_b_size,)) / source_b_size
    # generate n samples of alpha, the p-dim vector
    alpha = systematic_generate_alpha(n,dirichlet_dim,lbd,ubd,size = alpha_size, strata_nb = strata_nb, method = method)
    result = []
    for i in range(len(alpha)):
        dir_probability = dirichlet.rvs(alpha[i],size=(target_size))
        # unravel the ndarray
        rho_choice = random.choice(dir_probability.ravel(),size = (int(target_size*d))) # return type is numpy.ndarray
        # randomly generate signs
        signs = generate_signs(target_size*d)
        rho_choice = signs*rho_choice
        if d > 1:
            rho_choice = np.array(np.array_split(rho_choice, target_size, axis=0)) # list of target_size sub-arrays of d entries converted to array
        M2_a = cdist(a.reshape(source_a_size,d), rho_choice.reshape(target_size,d),lambda u, v: -np.dot(u,v)) # loss matrix       
        M2_b = cdist(b.reshape(source_b_size,d), rho_choice.reshape(target_size,d),lambda u, v: -np.dot(u,v)) # loss matrix       
        diff = -ot.emd2(x1, x2, M2_a)+ ot.emd2(x3,x2, M2_b)
        result.append(diff)
    myMin = np.array(result).min()
    return myMin



def generate_an_optimal_dirichlet_alternative(a,b,n = 1000,target_size = 100, lbd = 1, ubd = 101, alpha_size = 10, strata_nb = 2, method = 'random'):
    r""" Return the minimal Wasserstein distance under an optimal measure 'rho' 

    The main idea of this function is to calculate the minimal Wasserstein distance under an optimal measure 'rho'. Instead of using Dirichlet distribution to model the probability mass function for a general discrete probability measure 'rho', this method samples 'rho' directly through Dirichlet random variables. We allow for negative values by randomly generating signs to the Dirichlet random variables. To sample the Dirichlet random variables, this method simply selects the fist 'd' (where d = dimension) columns of the generated alphas.  
    
    Parameters
    ----------
    a : ndarray, float64
        ndarray of the first source distribution. It should be of shape(n,d), where d is the dimension. For all dimensions, 'a' is simply the ndarray of samples 
    b : ndarray, float64
        ndarray of the second source distribution. It should be of shape(n,d), where d is the dimension. For all dimensions, 'b is simply the ndarray of samples
    n : int
        number of the Dirichlet parameter alpha sampled
    target_size : int
        the number of bins (for d=1) or number samples (for d>1) requested for the optimal measure 'rho'
    lbd : float64
        the lower bound of the alpha parameter for Dirichlet distribution
    ubd : float64
        the upper bound of the alpha parameter for Dirichlet distribution
    alpha_size : int
        number of sub-parts whereby we partition the grid in each dimension of the alpha parameter
    strata_nb : int
        number of alpha entries we generate for each sub-part of the partitioned grid of the alpha parameter for each dimension
    method : str
        the method that we use to sample alpha. Legal names include: "random", "auto random strata probing", "auto random strata probing zip", "random strata probing", and "extrema strata probing"
               
    Returns
    -------
    myMin : float64
        a single float that refers to the minimal Wasserstein distance found across all sampled alpha parameters
    """
    # cast input into ndarray
    a = np.array(a)
    b = np.array(b)
    # check input validity wrt dimensions
    if a.ndim != b.ndim:
        raise ValueError("Your input source distributions should come from the same dimension!")
    d = a.ndim
    dirichlet_dim = d+1
    source_a_size = len(a)
    source_b_size = len(b)
    x1 = np.ones((source_a_size,)) / source_a_size # uniform distribution on samples
    x2 = np.ones((target_size,)) / target_size
    x3 = np.ones((source_b_size,)) / source_b_size
    # generate n samples of alpha, the p-dim vector
    alpha = systematic_generate_alpha(n,dirichlet_dim,lbd,ubd,size = alpha_size, strata_nb = strata_nb, method = method)
    result = []
    for i in range(len(alpha)):
        dir_probability = dirichlet.rvs(alpha[i],size=(target_size))
        # unravel the ndarray
        rho = dir_probability[:,:d].ravel() # rho_variable itself
        # randomly generate signs
        signs = generate_signs(target_size*d)
        rho = signs*rho
        if d > 1:
            rho = np.array(np.array_split(rho, target_size, axis=0))
        M2_a = cdist(a.reshape(source_a_size,d), rho.reshape(target_size,d),lambda u, v: -np.dot(u,v)) # loss matrix        
        M2_b = cdist(b.reshape(source_b_size,d), rho.reshape(target_size,d),lambda u, v: -np.dot(u,v)) # loss matrix        
        diff = -ot.emd2(x1, x2, M2_a)+ ot.emd2(x3,x2, M2_b)
        result.append(diff)
    myMin = np.array(result).min()
    return myMin


def generate_N_samples(N = 100, n = 1000,p = 10, a=gauss(100, m=20, s=5),b = gauss(100, m=10, s=1), a_grid = np.arange(100, dtype=np.float64), b_grid = np.arange(100, dtype=np.float64),target_size = 100,lbd = 1, ubd = 101, op_method = "Hist", alpha_size = 10, strata_nb = 2, method = 'random'):
    r""" Generate 'N' samples of minimal Wasserstein distance under an optimal measure 'rho' through a chosen method

    The main idea of this function is to generate 'N' samples of minimal Wasserstein distance under an optimal measure 'rho' through a chosen method. Allowed legal names are: "Hist" which calls function 'generate_an_optimal'; "Samples" which calls function 'generate_an_optimal_samples'; "Dirichlet Random" which calls function 'generate_an_optimal_dirichlet'; and "Dirichlet" which calls function 'generate_an_optimal_dirichlet_alternative'
    
    Parameters
    ----------
    N : int
        total number of samples of minimal Wasserstein distance
    n : int
        number of the Dirichlet parameter alpha sampled
    p : int
        number whereby we partition the grid for the support of the optimal measure 'rho'
    a : ndarray, float64
        ndarray of the first source distribution. It should be of shape(n,d), where d is the dimension. If 'Hist' is chosen, for d = 1, 'a' is the ndarray of sample weights (or histograms) of the source distribution. For d > 1, 'a' is simply the ndarray of samples. Otherwise, for all dimensions, 'a' is simply the ndarray of samples 
    b : ndarray, float64
        ndarray of the second source distribution. It should be of shape(n,d), where d is the dimension. If 'Hist' is chosen, for d = 1, 'b' is the ndarray of sample weights (or histograms) of the source distribution. For d > 1, 'b' is simply the ndarray of samples. Otherwise, for all dimensions, 'b' is simply the ndarray of samples 
    a_grid : ndarray, float64
        ndarray of the histogram grid for the first source distribution. This parameter is only required for d = 1
    b_grid : ndarray, float64
        ndarray of the histogram grid for the second source distribution. This parameter is only required for d = 1
    target_size : int
        the number of bins (for d=1) or number samples (for d>1) requested for the optimal measure 'rho'
    lbd : float64
        the lower bound of the alpha parameter for Dirichlet distribution
    ubd : float64
        the upper bound of the alpha parameter for Dirichlet distribution
    op_method : str
        specify the optimization method; see legal names above in the description
    alpha_size : int
        number of sub-parts whereby we partition the grid in each dimension of the alpha parameter
    strata_nb : int
        number of alpha entries we generate for each sub-part of the partitioned grid of the alpha parameter for each dimension
    method : str
        the method that we use to sample alpha. Legal names include: "random", "auto random strata probing", "auto random strata probing zip", "random strata probing", and "extrema strata probing"
               
    Returns
    -------
    mySamples : ndarray, float64
        1darray of floats that correspond to the sampled minimal Wasserstein distance
    """
    mySamples = []
    if op_method == "Hist":
        for i in range(N):
            mySamples.append(generate_an_optimal(n = n,p = p,a = a,b = b,a_grid = a_grid,b_grid=b_grid, target_size = target_size, lbd = lbd,ubd = ubd,alpha_size = alpha_size, strata_nb = strata_nb, method = method))
    elif op_method == "Samples":
        for i in range(N):
            mySamples.append(generate_an_optimal_samples(n = n,a = a,b = b,lbd = lbd,ubd = ubd,target_size = target_size, alpha_size = alpha_size, strata_nb = strata_nb, method = method))
    elif op_method == "Dirichlet Random":
        for i in range(N):
            mySamples.append(generate_an_optimal_dirichlet(a = a,b = b,n = n,target_size = target_size, lbd = lbd, ubd = ubd, alpha_size = alpha_size, strata_nb = strata_nb, method = method))
    elif op_method == "Dirichlet":
        for i in range(N):
            mySamples.append(generate_an_optimal_dirichlet_alternative(a = a,b = b,n = n,target_size = target_size, lbd = lbd, ubd = ubd, alpha_size = alpha_size, strata_nb = strata_nb, method = method))
    else:
        raise ValueError("Must specify optimization method: 'Hist', 'Samples',  'Dirichlet Random', or 'Dirichlet'.")
    return np.array(mySamples)



def sample_inference(N = 100, n = 1000,p = 10, a=gauss(100, m=20, s=5),b = gauss(100, m=10, s=1), a_grid = np.arange(100, dtype=np.float64), b_grid = np.arange(100, dtype=np.float64),target_size = 100,lbd = 1, ubd = 101, op_method = "Hist", alpha_size = 10, strata_nb = 2, method = 'random', alpha = 0.05):
    r""" Generate sample mean, standard deviation, confidence interval, and minimum for 'N' samples of minimal Wasserstein distance under an optimal measure 'rho' through a chosen method

    The main idea of this function is to generate sample mean, standard deviation, confidence interval, and minimum for 'N' samples of minimal Wasserstein distance under an optimal measure 'rho' through a chosen method. Allowed legal names are: "Hist" which calls function 'generate_an_optimal'; "Samples" which calls function 'generate_an_optimal_samples'; "Dirichlet Random" which calls function 'generate_an_optimal_dirichlet'; and "Dirichlet" which calls function 'generate_an_optimal_dirichlet_alternative'
    
    Parameters
    ----------
    N : int
        total number of samples of minimal Wasserstein distance
    n : int
        number of the Dirichlet parameter alpha sampled
    p : int
        number whereby we partition the grid for the support of the optimal measure 'rho'
    a : ndarray, float64
        ndarray of the first source distribution. It should be of shape(n,d), where d is the dimension. If 'Hist' is chosen, for d = 1, 'a' is the ndarray of sample weights (or histograms) of the source distribution. For d > 1, 'a' is simply the ndarray of samples. Otherwise, for all dimensions, 'a' is simply the ndarray of samples 
    b : ndarray, float64
        ndarray of the second source distribution. It should be of shape(n,d), where d is the dimension. If 'Hist' is chosen, for d = 1, 'b' is the ndarray of sample weights (or histograms) of the source distribution. For d > 1, 'b' is simply the ndarray of samples. Otherwise, for all dimensions, 'b' is simply the ndarray of samples 
    a_grid : ndarray, float64
        ndarray of the histogram grid for the first source distribution. This parameter is only required for d = 1
    b_grid : ndarray, float64
        ndarray of the histogram grid for the second source distribution. This parameter is only required for d = 1
    target_size : int
        the number of bins (for d=1) or number samples (for d>1) requested for the optimal measure 'rho'
    lbd : float64
        the lower bound of the alpha parameter for Dirichlet distribution
    ubd : float64
        the upper bound of the alpha parameter for Dirichlet distribution
    op_method : str
        specify the optimization method; see legal names above in the description
    alpha_size : int
        number of sub-parts whereby we partition the grid in each dimension of the alpha parameter
    strata_nb : int
        number of alpha entries we generate for each sub-part of the partitioned grid of the alpha parameter for each dimension
    method : str
        the method that we use to sample alpha. Legal names include: "random", "auto random strata probing", "auto random strata probing zip", "random strata probing", and "extrema strata probing".
    alpha : float64
        specifies the (1-'alpha')% confidence level
               
    Returns
    -------
    return_result : list, various
        return_result[0]: float64
            sample mean
        return_result[1]: float64
            sample standard deviation
        return_result[2]: list, float64
            return_result[2][0]: lower bound of confidence interval
            return_result[2][1]: upper bound of confidence interval
        return_result[3]: float64
            the minimum across all minimal Wasserstein distance sampled     
    """
    mySample = generate_N_samples(N,n,p,a,b,a_grid,b_grid,target_size,lbd,ubd,op_method,alpha_size,strata_nb,method)
    sm = mySample.mean()
    sd = math.sqrt(mySample.var()/N)
    lts = stats.norm.ppf(alpha/2,loc = 0, scale = 1)
    uts = stats.norm.ppf(1-alpha/2,loc = 0, scale = 1)
    confidence_interval = [sm-uts*sd,sm-lts*sd]
    Min = mySample.min()
    return_result = [sm,sd,confidence_interval,Min]
    return return_result


def plot_1D_rho(p,opt_rho):
    r""" prints the histogram of a discrete probability distribution 'rho'

    The main idea of this function is to print the 1D discrete optimal probability method 'rho' sampled through 'Histogram' method
    
    Parameters
    ----------
    p : int
        number whereby we partition the grid for the support of the optimal measure 'rho'
    opt_rho : ndarray
        ndarray of point-wise probability with respect to each point in the corresponding support
               
    Returns
    -------
    void
    
    prints graphs as side effects
    """
    xk = np.arange(p)
    custm = stats.rv_discrete(name='custm', values=(xk, opt_rho))
    fig, ax = plt.subplots(1, 1)
    ax.plot(xk, custm.pmf(xk), 'ro', ms=12, mec='r')
    ax.vlines(xk, 0, custm.pmf(xk), colors='r', lw=4)
    # change size of plot
    plt.gcf().set_size_inches(12, 10)
    plt.show()

def plot_2D_rho(a,b,opt_rho):
    r"""prints the graphs of samples from source_a distribution and source_b distribution wrt probability measure 'opt_rho' respectively in 2D space

    The main idea of this function is to print the graphs of samples from source_a distribution and source_b distribution wrt probability measure 'opt_rho' respectively in 2D space through 'Samples', 'Dirichlet Random', and 'Dirichlet' method
    
    Parameters
    ----------
    a : ndarray, float64
        ndarray of the first source distribution. It should be of shape(n,d), where d is the dimension. For all dimensions, 'a' is simply the ndarray of samples 
    b : ndarray, float64
        ndarray of the second source distribution. It should be of shape(n,d), where d is the dimension. For all dimensions, 'b' is simply the ndarray of samples 
    opt_rho : ndarray
        ndarray of 2D samples generated from probability distribution 'rho'.
               
    Returns
    -------
    void
    
    prints graphs as side effects
    
    Reference
    ---------
    code is adapted from POT example "Optimal Transport between 2D empirical distributions"
    """
    pl.figure(1)
    pl.plot(a[:, 0], a[:, 1], '+b', label='Source samples')
    pl.plot(opt_rho[:, 0], opt_rho[:, 1], 'xr', label='Target samples')
    pl.legend(loc=0)
    plt.gcf().set_size_inches(12, 10)
    pl.title('Source A and opt_rho distributions')
    pl.figure(2)
    pl.plot(b[:, 0], b[:, 1], '+b', label='Source samples')
    pl.plot(opt_rho[:, 0], opt_rho[:, 1], 'xr', label='Target samples')
    pl.legend(loc=0)
    plt.gcf().set_size_inches(12, 10)
    pl.title('Source B and opt_rho distributions')
    

def plot_2D_OTMatrix (a,b,opt_rho,a_grid,b_grid,opt_rho_grid,M_a,M_b):
    r""" prints the graphs of transport matrix from source_a distribution to 'rho' and source_b distribution to 'rho'

    The main idea of this function is to print the graphs of prints the graphs of transport matrix from source_a distribution to 'rho' and source_b distribution to 'rho' for 2D
    
    Parameters
    ----------
    a : ndarray, float64
        ndarray of the first source distribution. It should be of shape(n,d), where d is the dimension. For all dimensions, 'a' is simply the ndarray of samples 
    b : ndarray, float64
        ndarray of the second source distribution. It should be of shape(n,d), where d is the dimension. For all dimensions, 'b' is simply the ndarray of samples 
    opt_rho : ndarray
        ndarray of 2D samples generated from probability distribution 'rho'
    a_grid : ndarray, float64
        ndarray of sample weights for source_a distribution
    b_grid : ndarray, float64
        ndarray of sample weights for source_b distribution
    rho_grid : ndarray, float64
        ndarray of sample weights for rho
    M_a : ndarray, float64
        ndarray of squared euclidean distance from source_a distribution to rho
    M_b : ndarray, float64
        ndarray of squared euclidean distance from source_b distribution to rho
            
    Returns
    -------
    void
    
    prints graphs as side effects
    """
    
    G0 = ot.emd(a, opt_rho, M_a)
    pl.figure(1)
    pl.imshow(G0, interpolation='nearest')
    pl.title('OT matrix for Source_a and Opt_rho')
    pl.figure(2)
    ot.plot.plot2D_samples_mat(a_grid, rho_grid, G0, c=[.5, .5, 1])
    pl.plot(a_grid[:, 0], a_grid[:, 1], '+b', label='Source a samples')
    pl.plot(rho_grid[:, 0], rho_grid[:, 1], 'xr', label='Target (opt_rho) samples')
    pl.legend(loc=0)
    pl.title('OT matrix with samples: Source_a and Opt_rho')
    
    G1 = ot.emd(b, opt_rho, M_b)
    pl.figure(3)
    pl.imshow(G1, interpolation='nearest')
    pl.title('OT matrix for Source_a and Opt_rho')
    pl.figure(4)
    ot.plot.plot2D_samples_mat(b_grid, rho_grid, G1, c=[.5, .5, 1])
    pl.plot(b_grid[:, 0], b_grid[:, 1], '+b', label='Source b samples')
    pl.plot(rho_grid[:, 0], rho_grid[:, 1], 'xr', label='Target (opt_rho) samples')
    pl.legend(loc=0)
    pl.title('OT matrix with samples: Source_a and Opt_rho')
    


def plot_an_optimalMeasure(n = 1000,p = 10, a=gauss(100, m=15, s=5),b = gauss(100, m=10, s=1),a_grid = np.arange(100, dtype=np.float64), b_grid = np.arange(100, dtype=np.float64), target_size = 100, lbd = 1, ubd = 101,  alpha_size = 10, strata_nb = 2, method = 'random', op_method = "Hist"):
    r""" Returns optimal measure rho either in the form of histogram or samples and prints corresponding graphs as side effects

    The main idea of this function is to return to the users the optimal measure 'rho' and prints its histogram in 1D or both the sample distribution graphs displaying source vs target distribution and graph for the perspective transportation matrix in 2D
    
    Parameters
    ----------
    n : int
        number of the Dirichlet parameter alpha sampled
    p : int
        number whereby we partition the grid for the support of the optimal measure 'rho'
    a : ndarray, float64
        ndarray of the first source distribution. It should be of shape(n,d), where d is the dimension. If 'Hist' is chosen, for d = 1, 'a' is the ndarray of sample weights (or histograms) of the source distribution. For d > 1, 'a' is simply the ndarray of samples. Otherwise, for all dimensions, 'a' is simply the ndarray of samples 
    b : ndarray, float64
        ndarray of the second source distribution. It should be of shape(n,d), where d is the dimension. If 'Hist' is chosen, for d = 1, 'b' is the ndarray of sample weights (or histograms) of the source distribution. For d > 1, 'b' is simply the ndarray of samples. Otherwise, for all dimensions, 'b' is simply the ndarray of samples 
    a_grid : ndarray, float64
        ndarray of the histogram grid for the first source distribution. This parameter is only required for d = 1
    b_grid : ndarray, float64
        ndarray of the histogram grid for the second source distribution. This parameter is only required for d = 1
    target_size : int
        the number of bins (for d=1) or number samples (for d>1) requested for the optimal measure 'rho'
    lbd : float64
        the lower bound of the alpha parameter for Dirichlet distribution
    ubd : float64
        the upper bound of the alpha parameter for Dirichlet distribution
    alpha_size : int
        number of sub-parts whereby we partition the grid in each dimension of the alpha parameter
    strata_nb : int
        number of alpha entries we generate for each sub-part of the partitioned grid of the alpha parameter for each dimension
    method : str
        the method that we use to sample alpha. Legal names include: "random", "auto random strata probing", "auto random strata probing zip", "random strata probing", and "extrema strata probing".
    op_method : str
        specify the optimization method; see legal names above in the description
            
    Returns
    -------
    opt_rho : ndarray, float64
        ndarray of sample weights from 'rho' for 1D "Hist" method and ndarray of samples drawn from 'rho' for others
    
    prints graphs as side effects
    """
    source_a_size = len(a)
    source_b_size = len(b)
    # cast input into ndarray
    a = np.array(a)
    b = np.array(b)
    # check input validity wrt dimensions
    if a.ndim != b.ndim:
        raise ValueError("Your input source distributions should come from the same dimension!")
    d = a.ndim    
    # check that dimension is <=2
    if d > 2:
        raise ValueError("This function only supports dimension <= 2.")
    # check validity of input
    if source_a_size != len(a_grid) or source_b_size != len(b_grid):
        raise ValueError("Your source distribution size and its corresponding grid size do not match!")
    
    # convert passed-in grids to np.array
    a_grid = np.array(a_grid)
    b_grid = np.array(b_grid)
    
    # check validity of request
    
    if d != 1 and op_method == "Hist":
        raise ValueError("Histogram Method is only available for 1-dimensional distributions!")
        
    # generate n samples of alpha, the p-dim vector
    alpha = systematic_generate_alpha(n,rho_num,lbd,ubd,size = alpha_size, strata_nb = strata_nb, method = method)
    
    # if user chooses 'histogram' method
    if op_method == "Hist": 
        # (1). set up the grid 
        rho_grid = np.linspace(-1, 1, p) # target bin positions
        # (2). calculate cost matrixes
        M2_a = cdist(a_grid.reshape((source_a_size, 1)), rho_grid.reshape((p, 1)),  lambda u, v: -np.dot(u,v))
        M2_b = cdist(b_grid.reshape((source_b_size, 1)), rho_grid.reshape((p, 1)),  lambda u, v: -np.dot(u,v))
        result = []
        rho_list = []
        for i in range(n):
            dir_probability = dirichlet.rvs(alpha[i])
            # unravel the ndarray
            rho_rv = dir_probability.ravel()
            rho_list.append(rho_rv)
            diff = -ot.emd2(a, rho_rv, M2_a)+ ot.emd2(b, rho_rv, M2_b)
            result.append(diff)
        min_idx = result.index(min(result))
        opt_rho = rho_list[min_idx]
        # plot optimal rho
        plot_1D_rho(p,opt_rho)

        
    # set up for non-hist methods
    elif op_method == "Samples" or op_method == "Dirichlet Random" or op_method == "Dirichlet":
        x1 = np.ones((source_a_size,)) / source_a_size # uniform distribution on samples
        x2 = np.ones((target_size,)) / target_size
        x3 = np.ones((source_b_size,)) / source_b_size
        
        # 1. Samples Method
        if op_method == "Samples":
            rho_num = p**d # total number of categorical variables
            # d-D grid
            if d == 1:
                coordinates = np.linspace(-1, 1, p)        
            else:
                grid = np.array([np.linspace(-1, 1, p)])
                multi_grid = np.repeat(grid,d,axis=0)
                coordinates = np.array(np.meshgrid(*multi_grid)).T.reshape(-1, d) # ndarray of all the discrete points in the grid
            rho_list = []
            result = []
            M2_a_list = []
            M2_b_list = []
            for i in range(len(alpha)):
                dir_probability = dirichlet.rvs(alpha[i])
                # unravel the ndarray
                rho_rv = dir_probability.ravel() # discrete probability of each point of the grid
                xk = np.arange(rho_num)
                custm = stats.rv_discrete(name='custm', values=(xk, rho_rv))
                R = custm.rvs(size=target_size) # index array
                rho = coordinates[R]
                rho_list.append(rho)
                M2_a = cdist(a.reshape(source_a_size,d), rho.reshape(target_size,d),lambda u, v: -np.dot(u,v)) # loss matrix       
                M2_b = cdist(b.reshape(source_b_size,d), rho.reshape(target_size,d),lambda u, v: -np.dot(u,v)) # loss matrix       
                diff = -ot.emd2(x1, x2, M2_a)+ ot.emd2(x3,x2, M2_b)
                result.append(diff)
                M2_a_list.append(M2_a)
                M2_b_list.append(M2_b)
            min_idx = result.index(min(result))
            opt_rho = rho_list[min_idx]
            opt_M2_a = M2_a_list[min_idx]
            opt_M2_b = M2_b_list[min_idx]
            # plot optimal rho
            if d == 1:
                plot_1D_rho(p,opt_rho)
            else:
                # plot optimal rho
                plot_2D_rho(a,b,opt_rho)             
                # plot 2D OT (EMD) transport map
                plot_2D_OTMatrix(a,b,opt_rho,a_grid = x1,b_grid = x3,opt_rho_grid = x2,M_a = opt_M2_a,M_b = opt_M2_b)
                
           
        
        # 2. Dirichlet Methods
        elif op_method == "Dirichlet Random" or op_method == "Dirichlet":
            dirichlet_dim = d+1
            rho_list = []
            result = []
            M2_a_list = []
            M2_b_list = []
            for i in range(len(alpha)):
                dir_probability = dirichlet.rvs(alpha[i],size=(target_size))
                # randomly generate signs
                signs = generate_signs(target_size*d)
            
                # 2.1 Dirichlet Random Method
                if op_method == "Dirichlet Random":
                    rho_random = random.choice(dir_probability.ravel(),size = (int(target_size*d))) # return type is numpy.ndarray
                    rho_random = signs*rho_random
                    if d > 1:
                        rho_random = np.array(np.array_split(rho_random, target_size, axis=0)) # list of target_size sub-arrays of d entries converted to array
                    M2_a = cdist(a.reshape(source_a_size,d), rho_random.reshape(target_size,d),lambda u, v: -np.dot(u,v)) # loss matrix   
                    M2_b = cdist(b.reshape(source_b_size,d), rho_random.reshape(target_size,d),lambda u, v: -np.dot(u,v)) # loss matrix   
                    diff = -ot.emd2(x1, x2, M2_a)+ ot.emd2(x3,x2, M2_b)
                    result.append(diff)
                    rho_list.append(rho_random)
                
                # 2.2 Dirichlet Method
                else:
                    rho = dir_probability[:,:d].ravel() # rho_variable itself
                    rho = signs*rho
                    if d > 1:
                        rho = np.array(np.array_split(rho, target_size, axis=0))
                    M2_a = cdist(a.reshape(source_a_size,d), rho.reshape(target_size,d),lambda u, v: -np.dot(u,v)) # loss matrix   
                    M2_b = cdist(b.reshape(source_b_size,d), rho.reshape(target_size,d),lambda u, v: -np.dot(u,v)) # loss matrix   
                    diff = -ot.emd2(x1, x2, M2_a)+ ot.emd2(x3,x2, M2_b)
                    result.append(diff)
                    M2_a_list.append(M2_a)
                    M2_b_list.append(M2_b)
                    
            min_idx = result.index(min(result))
            opt_rho = rho_list[min_idx]
            opt_M2_a = M2_a_list[min_idx]
            opt_M2_b = M2_b_list[min_idx]
            # plot optimal rho
            if d == 1:
                plot_1D_rho(p,opt_rho)
            else:
                # plot optimal rho
                plot_2D_rho(a,b,opt_rho)             
                # plot 2D OT (EMD) transport map
                plot_2D_OTMatrix(a,b,opt_rho,a_grid = x1,b_grid = x3,opt_rho_grid = x2,M_a = opt_M2_a,M_b = opt_M2_b)
        
    return opt_rho    


