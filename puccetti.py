import numpy as np
from itertools import permutations
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, plotting
from scipy.stats import dirichlet
import math
from scipy import stats
from itertools import product
from scipy.spatial.distance import cdist
from numpy import random
import matplotlib.pylab as pl
import ot
import scipy as scipy
from ot.datasets import make_1D_gauss as gauss



def swapping_criterion(sigma, sigma_alt, a, b, epsilon):
    N = len(a)
    sum_ls = []
    for i in range(N):
        sum_ls.append(np.dot(a[i],b[sigma[i]])-np.dot(a[i],b[sigma_alt[i]]))
    result = np.abs(sum(sum_ls))/N
    if result < epsilon:
        return [True, result]
    else:
        return [False, result]
    
    
def swap(a,b,sigma):
    N = len(a)   
    sigma_alt = sigma # initiate sigma_alt for swapping
    for i in range(0, N):
        for j in range(i, N):
            if np.dot(a[i],b[sigma[i]])+np.dot(a[j],b[sigma[j]])> np.dot(a[i],b[sigma[j]])+np.dot(a[j],b[sigma[i]]):
                # swap
                temp = sigma_alt[i]
                sigma_alt[i] = sigma_alt[j]
                sigma_alt[j] = temp
    return sigma_alt


def swapping_algo(a,b,epsilon):
    N = len(a)
    sigma = list(range(N)) # the identity permutation
    sigma_alt = swap(a,b,sigma)
    while not swapping_criterion(sigma, sigma_alt, a, b, epsilon)[0]:
        sigma = sigma_alt
        sigma_alt = swap(a,b,sigma)
    sigma_star = sigma_alt
    sum_ls = []
    for i in range(N):
        sum_ls.append(np.dot(a[i],b[sigma_star[i]]))
    
    loss = (1/N)*sum(sum_ls)
    return sigma_star, loss


# below are the adjusted bayesian algorithm using puccetti swapping algorithm

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


def return_search_space(p,d):
    r"""Generate search space for alpha
    The main idea of this function is to iteratively define a search space for multidimensional vector alpha
    Parameters
    ----------
    p : int
        partition size of the sample space
    d: int
        dimension of the source sample
        
    Returns
    -------
    my_dict : dictionary
        a dictionary defining the search space for vector alpha
    """
    my_dict = {}
    for i in range(p**d):
        my_dict["alpha_"+str(i)] = hp.uniform("alpha_"+str(i), 1, 101)
    return my_dict


def bayesian_optimization(method, a, b, plot = False, p=5, lbd = 1, ubd = 101, epsilon = 0.000001, algo = tpe.suggest, max_eval = 300, as_dict = False):
    r"""provides bayesian optimization with respect to each provided modelling methods.
    The main idea of this function is to provide bayesian optimization with respect to each provided modelling methods.
    Parameters
    ----------
    method: str
        method can be "sample," "hist," or "dir," each representing a modeling method for rho
    a : ndarray, float64
        ndarray of the first source distribution. It should be of shape(n,d), where d is the dimension. For d = 1, 'a' is the ndarray of sample weights (or histograms) of the source distribution. For d > 1, 'a' is simply the ndarray of samples 
    b : ndarray, float64
        ndarray of the second source distribution. It should be of shape(n,d), where d is the dimension. For d = 1, 'b' is the ndarray of sample weights (or histograms) of the source distribution. For d > 1, 'b' is simply the ndarray of samples
    a_grid : ndarray, float64
        ndarray of the histogram grid for the first source distribution. This parameter is only required for method == 'hist'
    b_grid : ndarray, float64 
        ndarray of the histogram grid for the second source distribution. This parameter is only required for method == 'hist'
    plot: bool
        chooses to plot (or not plot) the hyperopt graph
    p : int
        number whereby we partition the grid for the support of the optimal measure 'rho' 
    target_size : int
        the number of bins (for d=1) or number samples (for d>1) requested for the optimal measure 'rho'
    lbd : float64
        the lower bound of the alpha parameter for Dirichlet distribution
    ubd : float64
        the upper bound of the alpha parameter for Dirichlet distribution
    algo : function
        a function to select the optimization method in the hyperopt package
    max_eval: int
        maximum number of evaluation for the bayesian optimization per trial
    as_dict: bool
        True if the required value is returned as a dictionary, False if required value is returned as a list of lists containing the minimal Wasserstein distance and the optimizing parameter rho.
        
    Returns
    -------
    best : dictionary (if as_dict == True)
        a dictionary containing the optimizing rho
    result : list, list, float64 (if as_dict == False)
        a list of lists containing the minimal Wasserstein distance and the optimizing parameter alpha
    """    
    global size
    size = len(a)
    global mu
    mu = np.array(a)
    global nu 
    nu = np.array(b)
    global d
    d = a.ndim
    global partition
    partition = p
    global eps
    eps = epsilon
    if method == "samples":
        return hyperopt_samples(lbd, ubd, algo, max_eval, plot, as_dict)
    elif method == "dir":
        return hyperopt_dir(lbd, ubd, algo, max_eval, plot, as_dict)
    else:
        raise Exception("Please choose a valid method : 'samples', 'hist', or 'dir' . ")

        
def hyperopt_samples(lbd = 1, ubd = 101, algo = tpe.suggest, max_eval = 300, plot = False, as_dict = False):
    r"""hyperopt method for the "Indirect dirichlet with samples" model
    The main idea of this function is to return the appropriate hyperopt fmin solver for the "Indirect dirichlet with samples" model.
    Parameters
    ----------
    lbd : float64
        the lower bound of the alpha parameter for Dirichlet distribution
    ubd : float64
        the upper bound of the alpha parameter for Dirichlet distribution
    algo : function
        a function to select the optimization method in the hyperopt package
    max_eval: int
        maximum number of evaluation for the bayesian optimization per trial
    plot: bool
        choose whether to plot the hyperopt graphs or not       
    as_dict: bool
        True if the required value is returned as a dictionary, False if required value is returned as a list of lists containing the minimal Wasserstein distance and the optimizing parameter rho.
        
    Returns
    -------
    best : dictionary (if as_dict == True)
        a dictionary containing the optimizing rho
    result : list, list, float64 (if as_dict == False)
        a list of lists containing the minimal Wasserstein distance and the optimizing parameter rho
    """    
    trials = Trials()
    space = return_search_space(partition,d)
    best = fmin(fn = wasserstein_dist_diff_samples, space = space, algo = algo, trials = trials, max_evals = max_eval)
    if plot:
        plotting.main_plot_history(trials)
        plotting.main_plot_histogram(trials)
    if as_dict:
        return best
    else:
        best_loss = min(trials.losses())
        alpha_dict = list(best.items())
        best_alpha = []
        for i in range(len(alpha_dict)):
            best_alpha.append(alpha_dict[i][1])
        result = [[best_loss],best_alpha]
        return result


def hyperopt_dir(lbd = 1, ubd = 101, algo = tpe.suggest, max_eval = 300, plot = False, as_dict = False):
    r"""hyperopt method for the "direct dirichlet with randomization" model
    The main idea of this function is to return the appropriate hyperopt fmin solver for the "direct dirichlet with randomization" model.
    Parameters
    ----------
    lbd : float64
        the lower bound of the alpha parameter for Dirichlet distribution
    ubd : float64
        the upper bound of the alpha parameter for Dirichlet distribution
    algo : function
        a function to select the optimization method in the hyperopt package
    max_eval: int
        maximum number of evaluation for the bayesian optimization per trial
    plot: bool
        choose whether to plot the hyperopt graphs or not
    as_dict: bool
        True if the required value is returned as a dictionary, False if required value is returned as a list of lists containing the minimal Wasserstein distance and the optimizing parameter rho.
        
    Returns
    -------
    best : dictionary (if as_dict == True)
        a dictionary containing the optimizing rho
    result : list, list, float64 (if as_dict == False)
        a list of lists containing the minimal Wasserstein distance and the concentration parameter alpha controlling the respective optimizing dirichlet distribution rho
    """ 
    dirichlet_dim = d+1
    trials = Trials()
    space = return_search_space(partition,d = dirichlet_dim)
    best = fmin(fn = wasserstein_dist_diff_dir, space = space, algo = algo, trials = trials, max_evals = max_eval)
    if plot:
        plotting.main_plot_history(trials)
        plotting.main_plot_histogram(trials)
    if as_dict:
        return best
    else:
        best_loss = min(trials.losses())
        best_rho = trials.results[np.argmin([r['loss'] for r in trials.results])]['my_param']
        result = [[best_loss],best_rho]
        return result

def wasserstein_dist_diff_samples(params):
    r"""hyperopt objective function for the "Indirect dirichlet with samples" model
    The main idea of this function is to return the appropriate hyperopt objective function for the "Indirect dirichlet with samples" model.
    Parameters
    ----------
    params : list, float64
        the list of alphas selected from the search space
    
    Returns
    -------
    diff : float64
        the minimal Wasserstein distance
    """ 
    # 1. find rho
    alpha = []
    for i in range(partition**d):
        alpha.append(params["alpha_"+str(i)])
    alpha = np.array(alpha)
    rho_num = partition**d # total number of categorical variables
    if d == 1:
        coordinates = np.linspace(-1, 1, partition)        
    else:
        grid = np.array([np.linspace(-1, 1, partition)])
        multi_grid = np.repeat(grid,d,axis=0)
        coordinates = np.array(np.meshgrid(*multi_grid)).T.reshape(-1, d)  
    dir_probability = dirichlet.rvs(alpha)
    # unravel the ndarray
    rho_rv = dir_probability.ravel() # discrete probability of each point of the grid
    xk = np.arange(rho_num)
    custm = stats.rv_discrete(name='custm', values=(xk, rho_rv))
    R = custm.rvs(size=size) # index array
    rho = coordinates[R]
    
    # 2. apply puccetti
    diff_mu_rho = swapping_algo(mu,rho,eps)[1]
    diff_rho_nu = swapping_algo(rho,nu,eps)[1]
       
    diff = diff_mu_rho-diff_rho_nu
    return diff



def wasserstein_dist_diff_dir(params):
    r"""hyperopt objective function for the "Direct dirichlet with Randomization" model
    The main idea of this function is to return the appropriate hyperopt objective function for the "Direct dirichlet with Randomization" model.
    Parameters
    ----------
    params : list, float64
        the list of alphas selected from the search space
    
    Returns
    -------
    - : dictionary
        dictionary containing the minimal Wasserstein distance, the status object, and the optimizing vector rho
    """
    # 1. find rho
    alpha = []
    for i in range(partition):
        alpha.append(params["alpha_"+str(i)])
    alpha = np.array(alpha)
    dir_probability = dirichlet.rvs(alpha,size=(size))
    rho_choice = random.choice(dir_probability.ravel(),size = (int(size*d))) # return type is numpy.ndarray
    # randomly generate signs
    signs = generate_signs(size*d)
    rho_choice = signs*rho_choice
    if d > 1:
        rho_choice = np.array(np.array_split(rho_choice, size, axis=0)) # list of target_size sub-arrays of d entries converted to array
    # 2. apply puccetti
    diff_mu_rho = swapping_algo(mu,rho_choice,eps)[1]
    diff_rho_nu = swapping_algo(rho_choice,nu,eps)[1]
       
    diff = diff_mu_rho-diff_rho_nu
    return {'loss': diff, 'status': STATUS_OK,'my_param': rho_choice}


def get_opt_rho_1D(opt_alpha, ts):
    p = len(opt_alpha)
    dir_probability = dirichlet.rvs(opt_alpha)
    rho_rv = dir_probability.ravel()
    rho_samples = []
    for i in range(ts):
        rho_samples.append(np.random.choice(np.linspace(-1, 1, p), p=rho_rv))
    return [rho_rv,rho_samples]

def get_opt_rho_multiD(opt_alpha,d,ts):
    p = int(len(opt_alpha)**(1/d))
    dir_probability = dirichlet.rvs(opt_alpha)
    rho_rv = dir_probability.ravel()
    rho_index = []
    X = np.linspace(-1,1,p)
    Y = np.linspace(-1,1,p)
    ls = list(product(X,Y))
    for i in range(ts):
        rho_index.append(np.random.choice(np.arange(len(ls)), p=rho_rv))
    rho_samples = [list(ls[i]) for i in rho_index]
    return [rho_rv,rho_samples]


# tools in addition
def find_permutations(N):
    nums = list(range(1, N+1))
    perm_set = set(permutations(nums))
    return perm_set

def generate_pairs(N):
    pairs = []
    for i in range(0, N):
        for j in range(i, N):
            pairs.append((i, j))
    return pairs
