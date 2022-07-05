# Authors: Authors: Johannes Wiesel, Erica Zhang
# Version: June 30, 2022

# DESCRIPTION: This package includes all sampling methods to generate alpha that we use to sample the corresponding Dirichlet distribution in the optimization algorithm.

from partition_tools import *
import numpy as np
from scipy import stats
from itertools import product
import scipy as scipy
import math
from numpy import random

def random_generate_alpha(n, d, lbd = 0, ubd = 100):
    r"""Generate 'n' d-dimensional alphas randomly

    The main idea of this function is to randomly generate 'n' d-dimensional alpha parameter (i.e. as if drawing from a uniform distribution) within range [lbd, ubd] (inclusive) for dirichlet distribution sampling. 

    Parameters
    ----------
    n : int
        number of alpha
    d : int
        dimension of the alpha parameter
    lbd : float64
        lower bound of each entry of the alpha parameter
     ubd : float64
        upper bound of each entry of the alpha parameter   

    Returns
    -------
    alpha : list, list, float64
        list of length 'n' where each entry is a list of length 'd,' representing each d-dimensional alpha generated
    """
    alpha = []
    for i in range(n):
        alpha.append(random.randint(lbd, ubd, size=(d)))
    return alpha



def random_strata_probingBySize_mesh(n, d, lbd = 1, ubd = 101, size = 10):
      r"""Systematically generate 'n' d-dimensional alphas by probing linearly by size 'size' for each dimension then compose the generated alpha entries through permutation.

    The main idea of this function is to systematically generate 'n' d-dimensional alpha parameter so that we effectively sample the entire sample space for alpha. We first partition the grid for alpha to 'size' even sub-parts. We then evenly distribute the total number of alpha needed to be generated, i.e. 'n', to each sub-part. Within each sub-part, we randomly generate the perspective number of alpha parameters. Finally, we create a list of every possible permutation of the alpha entries in each dimension. Together, this gives the final list of alpha parameters sampled. 
    
    concatenate all alphas generated and return the composite result. 

    Parameters
    ----------
    n : int
        number of alpha
    d : int
        dimension of the alpha parameter
    lbd : float64
        lower bound of each entry of the alpha parameter
     ubd : float64
        upper bound of each entry of the alpha parameter 
    size : int
        number of sub-parts whereby we partition the grid in each dimension of the alpha parameter

    Returns
    -------
    alpha_ls : list, list, float64
        list of length math.ceil(n**(1/d))**d where each entry is a list of length 'd,' representing each d-dimensional alpha generated
    """
    count = d
    # discretize grid for each dimension of alpha
    x = np.arange(lbd,ubd).tolist()
    strata = even_slice_list(x,size) # grid in each dimension is discretized in to *size* even parts
    # distribute total nb of random samples evenly to stratas 
    dim_num = math.ceil(n**(1/d)) # number of samples for each dimension, if distributed evenly
    if dim_num <= 2:
        print("Note: math.ceil(n**(1/d)) <=2. random_sampling is invoked.",n, "alpha samples are generated. To stop this, increase n or decrease d.")
        return random_generate_alpha(n,d,lbd,ubd) # if nb of alpha samples generated in each dimension is too small, it is equivalent to random sampling
    strata_num_ls = distribute(dim_num, size)
    # because usually dimension will be very large, dim_num is close to 1, to optimize the algrorithm, we extract only the idices of nonzero elements
    nonzero_ls = list(np.array(strata_num_ls).nonzero())[0]
    # create a list of lists of samples of alpha generated for each dimension from 1 to d-1
    dim_ls = []
    while count > 0:
        sub_alpha_ls = []
        for i in range(len(nonzero_ls)):
            sub_alpha_ls.append(random.randint(strata[nonzero_ls[i]][0], strata[nonzero_ls[i]][size-1], size=(strata_num_ls[nonzero_ls[i]])).tolist())
        sub_alpha_ls = sum(sub_alpha_ls,[])
        count -= 1
        dim_ls.append(sub_alpha_ls)
    alpha_ls = list(product(*dim_ls))
    print("Note:",len(alpha_ls), "alpha samples are generated.")
    return alpha_ls



def random_strata_probingBySize_zip(n, d, lbd = 1, ubd = 101, size = 10):
     r"""Systematically generate 'n' d-dimensional alphas by probing linearly by size 'size' for each dimension then compose the generated alpha entries through zipping the entries together.

    The main idea of this function is to systematically generate 'n' d-dimensional alpha parameter so that we effectively sample the entire sample space for alpha. We first partition the grid for alpha to 'size' even sub-parts. We then evenly distribute the total number of alpha needed to be generated, i.e. 'n', to each sub-part. Within the sub-parts, we randomly generate the perspective number of alpha parameters. Finally, we create a list of d-dim alpha paramters through directly zipping together the generated alpha entries in each dimension. Together, this gives the final list of alpha parameters sampled. 
    This method differs from the previous method in that instead of producing the final return list of alpha parameters through permutation, we simply zip the generated alpha entry in each dimension together. While this potentially reduce the time complexity of this algorithm, it is inferior at sampling the alpha parameters, since the dirichlet distribution does not only depend on the relative magnitude of alpha, but also the proportional magnitude difference in entries of the alpha parameter.

    Parameters
    ----------
    n : int
        number of alpha
    d : int
        dimension of the alpha parameter
    lbd : float64
        lower bound of each entry of the alpha parameter
     ubd : float64
        upper bound of each entry of the alpha parameter 
    size : int
        number of sub-parts whereby we partition the grid in each dimension of the alpha parameter

    Returns
    -------
    alpha_ls : list, list, float64
        list of length math.ceil(n**(1/d))**d where each entry is a list of length 'd,' representing each d-dimensional alpha generated
    """
    count = d
    # discretize grid for each dimension of alpha
    x = np.arange(lbd,ubd).tolist()
    strata = even_slice_list(x,size) # grid in each dimension is discretized in to *size* even parts
    strata_num_ls = distribute(n, size)
    dim_ls = []
    while count > 0:
        sub_alpha_ls = []
        for i in range(size):
            sub_alpha_ls.append(random.randint(strata[i][0], strata[i][size-1], size=(strata_num_ls[i])).tolist())
        sub_alpha_ls = sum(sub_alpha_ls,[])
        count -= 1
        dim_ls.append(sub_alpha_ls)
    alpha_ls = list(zip(*dim_ls))
    print("Note:",len(alpha_ls), "alpha samples are generated.")
    return alpha_ls



def random_strata_probing_mesh(strata_nb, d, lbd = 1, ubd = 101, size = 10):
    r"""Generate 'strata_nb**d' d-dimensional alphas randomly

    The main idea of this function is to systematically generate 'n' d-dimensional alpha parameter so that we effectively sample the entire sample space for alpha. We first partition the grid for alpha to 'size' even sub-parts. Within each sub-part, we randomly generate the perspective number of alpha parameters. Finally, we create a list of every possible permutation of the alpha entries in each dimension. Together, this gives the final list of alpha parameters sampled.  Unlike function 'random_strata_probingBySize_mesh(n, d, lbd = 1, ubd = 101, size = 10)', this method takes in "strata_nb" rather than 'n' as parameter. We take in 'strata_nb' as a parameter to directly control for how many alpha entries in each dimension we would like to sample for for each sub-part of the partitioned grid. 

    Parameters
    ----------
    strata_nb : int
        number of alpha entries we generate for each sub-part of the partitioned grid for each dimension
    d : int
        dimension of the alpha parameter
    lbd : float64
        lower bound of each entry of the alpha parameter
    ubd : float64
        upper bound of each entry of the alpha parameter 
    size : int
        number of sub-parts whereby we partition the grid in each dimension of the alpha parameter


    Returns
    -------
    alpha_ls : list, list, float64
        list of length 'strata_nb**d' where each entry is a list of length 'd,' representing each d-dimensional alpha generated
    """
    count = d
    # discretize grid for each dimension of alpha
    x = np.arange(lbd,ubd).tolist()
    strata = even_slice_list(x,size) # grid in each dimension is discretized in to *size* even parts
    dim_ls = []
    while count > 0:
        sub_alpha_ls = []
        for j in range(size):
            sub_alpha_ls.append(random.randint(strata[j][0], strata[j][size-1], size=(strata_nb)).tolist())
        sub_alpha_ls = sum(sub_alpha_ls,[])
        count -= 1
        dim_ls.append(sub_alpha_ls)
    alpha_ls = list(product(*dim_ls))
    print("Note:",len(alpha_ls), "alpha samples are generated.")
    return alpha_ls



def extrema_strata_probing(size, d, lbd = 1, ubd = 101):  
    r"""Generate 'size**d' d-dimensional alphas by sampling only the boundary points of the partitioned grid for alpha

    The main idea of this function is to systematically generate 'size**d' d-dimensional alpha parameter so that we effectively sample the entire sample space for alpha. We first partition the grid for alpha to 'size' even sub-parts. Within the sub-parts, we select only the boundary points as entries for alpha. Finally, we create a list of every possible permutation of the alpha entries in each dimension. Together, this gives the final list of alpha parameters sampled.

    Parameters
    ----------
    size : int
        number of sub-parts whereby we partition the grid in each dimension of the alpha parameter
    d : int
        dimension of the alpha parameter
    lbd : float64
        lower bound of each entry of the alpha parameter
    ubd : float64
        upper bound of each entry of the alpha parameter 

    Returns
    -------
    alpha_ls : list, list, float64
        list of length 'strata_nb**d' where each entry is a list of length 'd,' representing each d-dimensional alpha generated
    """
    extrema = extreme_points(lbd,ubd,size)
    extreme_grid = np.repeat(np.array([extrema]),d,axis=0)
    alpha_ls = list(product(*extreme_grid))
    print("Note:",len(alpha_ls), "alpha samples are generated.")
    return alpha_ls



def systematic_generate_alpha(n = 500, d = 1, lbd = 1, ubd = 101, size = 10, strata_nb = 2, method = 'random'):
    r"""Generate 'n' d-dimensional alpha based on a method of choice

    The main idea of this function is to systematically generate 'n' d-dimensional alpha parameter based on a method of choice. Legal methods include "random", "auto random strata probing", "auto random strata probing zip", "random strata probing", extrema strata probing

    Parameters
    ----------
    n : int
        number of alpha
    d : int
        dimension of the alpha parameter
    lbd : float64
        lower bound of each entry of the alpha parameter
    ubd : float64
        upper bound of each entry of the alpha parameter 
    size : int
        number of sub-parts whereby we partition the grid in each dimension of the alpha parameter
    strata_nb : int
        number of alpha entries we generate for each sub-part of the partitioned grid for each dimension. This is requested if we choose "random strata probing."
 

    Returns
    -------
    This function calls other functions listed above. These functions all return: 
    
    list, list, float64:
        list of length 'n' where each entry is a list of length 'd,' representing each d-dimensional alpha generated
    """
    if method == 'random':
        return random_generate_alpha(n,d,lbd,ubd)
    elif method == 'auto random strata probing':
        return random_strata_probingBySize_mesh(n, d, lbd, ubd, size)
    elif method == 'auto random strata probing zip':
        return random_strata_probingBySize_zip(n, d, lbd, ubd, size)
    elif method == 'random strata probing':
        return random_strata_probing_mesh(strata_nb,d,lbd,ubd,size)
    elif method == 'extrema strata probing':
        return extrema_strata_probing(n, d, lbd, ubd)

    
