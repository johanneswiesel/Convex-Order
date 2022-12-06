import numpy as np
import matplotlib.pylab as pl
import ot
import scipy as scipy
from ot.datasets import make_1D_gauss as gauss
from scipy.stats import dirichlet
import math
from scipy import stats
from itertools import product
from scipy.spatial.distance import cdist
from numpy import random
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, plotting

# Authors: Johannes Wiesel, Erica Zhang
# Version: September 30, 2022

# DESCRIPTION: This package provides bayesian optimization tools to generate the minimal Wasserstein distance under an optimal measure 'rho'

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


def bayesian_optimization(method, a, b, a_grid = np.arange(100, dtype=np.float64), b_grid = np.arange(100, dtype=np.float64), plot = False, p=5, target_size = 100, lbd = 1, ubd = 101, algo = tpe.suggest, max_eval = 300, as_dict = False):
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
    # initialize static parameters
    a_grid = np.array(a_grid)
    b_grid = np.array(b_grid)
    global source_a_size
    source_a_size = len(a)
    global source_b_size
    source_b_size = len(b)
    global mu
    mu = np.array(a)
    global nu 
    nu = np.array(b)
    global d
    d = a.ndim
    global partition
    partition = p
    global ts
    ts = target_size
    global mu_grid
    mu_grid = a_grid
    global nu_grid
    nu_grid = b_grid
    if method == "samples":
        return hyperopt_samples(lbd, ubd, algo, max_eval, plot, as_dict)
    elif method == "hist":
        return hyperopt_hist(lbd, ubd, algo, max_eval, plot, as_dict)
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

def hyperopt_hist(lbd = 1, ubd = 101, algo = tpe.suggest, max_eval = 300, plot = False, as_dict = False):
    r"""hyperopt method for the "Indirect dirichlet with histogram" model
    The main idea of this function is to return the appropriate hyperopt fmin solver for the "Indirect dirichlet with histogram" model.
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
    trials = Trials()
    space = return_search_space(partition,d=1)
    best = fmin(fn = wasserstein_dist_diff_hist, space = space, algo = algo, trials = trials, max_evals = max_eval)
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
    # create a list of the alpha params
    alpha = []
    for i in range(partition**d):
        alpha.append(params["alpha_"+str(i)])
    alpha = np.array(alpha)
    x1 = np.ones((source_a_size,)) / source_a_size # uniform distribution on samples (array of bin heights rather than bin positions!)
    x2 = np.ones((ts,)) / ts
    x3 = np.ones((source_b_size,)) / source_b_size
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
    R = custm.rvs(size=ts) # index array
    rho = coordinates[R]
    M2_a = cdist(mu.reshape(source_a_size,d), rho.reshape(ts,d),lambda u, v: -np.dot(u,v)) # loss matrix        
    M2_b = cdist(nu.reshape(source_b_size,d), rho.reshape(ts,d),lambda u, v: -np.dot(u,v)) # loss matrix        
    diff = ot.emd2(x1, x2, M2_a)-ot.emd2(x3,x2, M2_b)
    return diff


def wasserstein_dist_diff_hist(params):
    r"""hyperopt objective function for the "Indirect dirichlet with histogram" model
    The main idea of this function is to return the appropriate hyperopt objective function for the "Indirect dirichlet with histogram" model.
    Parameters
    ----------
    params : list, float64
        the list of alphas selected from the search space
    
    Returns
    -------
    diff : float64
        the minimal Wasserstein distance
    """     
    # create a list of the alpha params
    alpha = []
    for i in range(partition):
        alpha.append(params["alpha_"+str(i)])
    alpha = np.array(alpha)
    rho_grid = np.linspace(-1, 1, partition) # target bin positions
    M2_a = cdist(mu_grid.reshape((source_a_size, 1)), rho_grid.reshape((partition, 1)),  lambda u, v: -np.dot(u,v))
    M2_b = cdist(nu_grid.reshape((source_b_size, 1)), rho_grid.reshape((partition, 1)),  lambda u, v: -np.dot(u,v))
    dir_probability = dirichlet.rvs(alpha)
    # unravel the ndarray
    rho_rv = dir_probability.ravel()
    diff = ot.emd2(mu, rho_rv, M2_a)-ot.emd2(nu, rho_rv, M2_b)
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
    # create a list of the alpha params
    alpha = []
    for i in range(partition):
        alpha.append(params["alpha_"+str(i)])
    alpha = np.array(alpha)
    x1 = np.ones((source_a_size,)) / source_a_size # uniform distribution on samples
    x2 = np.ones((ts,)) / ts
    x3 = np.ones((source_b_size,)) / source_b_size
    dir_probability = dirichlet.rvs(alpha,size=(ts))
    rho_choice = random.choice(dir_probability.ravel(),size = (int(ts*d))) # return type is numpy.ndarray
    # randomly generate signs
    signs = generate_signs(ts*d)
    rho_choice = signs*rho_choice
    if d > 1:
        rho_choice = np.array(np.array_split(rho_choice, ts, axis=0)) # list of target_size sub-arrays of d entries converted to array
    M2_a = cdist(mu.reshape(source_a_size,d), rho_choice.reshape(ts,d),lambda u, v: -np.dot(u,v)) # loss matrix       
    M2_b = cdist(nu.reshape(source_b_size,d), rho_choice.reshape(ts,d),lambda u, v: -np.dot(u,v)) # loss matrix       
    diff = ot.emd2(x1, x2, M2_a)-ot.emd2(x3,x2, M2_b)
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