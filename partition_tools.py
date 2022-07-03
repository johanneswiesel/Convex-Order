# Authors: Authors: Johannes Wiesel, Erica Zhang
# Version: June 30, 2022

# DESCRIPTION: This package provides helper tools for partitioning a list in different ways.

# Import dependencies 
import numpy as np
import scipy as scipy
import math


def distribute(oranges, plates):
    r"""Distribute oranges evenly among plates

    The main idea of this function is to distribute a given integer number(organges) evenly on s slots (plates) where the extra are randomly distributed.

    Parameters
    ----------
    oranges : int
        number of oranges to be evenly distributed
    plates : int
        number of plates to hold the oranges
    s : positive float64
        standard deviation

    Returns
    -------
    L : list, int
        list of length = plates where each entry corresponds to the number of oranges distributed to the plate at the idex
    """
    if plates == 0:
        raise ValueException('Number of plates must be >1!')
    base, extra = divmod(oranges, plates) # extra < plates
    L = [base for _ in range(plates)] # base distribution
    idx_ls = []
    if extra == 0:
        return L
    else:
        while extra>0:
            idx = scipy.stats.randint.rvs(0, plates) # lower bdd inclusive, upper bdd non-inclusive
            if idx not in idx_ls:
                idx_ls.append(idx)
                extra -= 1
        for i in range(len(idx_ls)):
            L[idx_ls[i]] += 1
        return L       
    

# partitioning a list into n equal parts 
# even distribution method

def even_slice_list(input_ls, size):
    r"""Return a list sliced into 'size' equal parts

    The main idea of this function is to partition a list into 'size' equal parts. This function uses 'distribution' as a helper function and assigns the extra randomly 

    Parameters
    ----------
    input_ls : list, float64
        the list that is to be sliced evenly
    size : int
        the requested number of parts the list is to be evenly sliced

    Returns
    -------
    return_ls : list, list, float64
        partitioned list of length 'size' of lists
    """
    input_size = len(input_ls)
    slice_ls = distribute(input_size,size)
    return_ls = []
    for i in range(size):
        ubd = sum(slice_ls[:i+1])
        lbd = sum(slice_ls[:i])
        return_ls.append(input_ls[lbd:ubd])
    return return_ls


# partitioning a list into n equal parts 
# consecutive division method

def slice_list(input_ls, size):
    r"""Return a list sliced into 'size' equal parts

    The main idea of this function is to partition a list into 'size' equal parts. This function assigns equal parts in a consecutive manner.

    Parameters
    ----------
    input_ls : list, float64
        the list that is to be sliced evenly
    size : int
        the requested number of parts the list is to be evenly sliced

    Returns
    -------
    return_ls : list, list, float64
        partitioned list of lists
    """
    input_size = len(input_ls)
    extra = input_size%size
    return_ls = []
    if extra != 0:
        base = int((input_size-extra)/(size-1))
        slice_ls = np.repeat([base],size-1,axis=0).tolist()
        #slice_ls = [int(x) for x in slice_ls] # convert all floats to int
        slice_ls.append(extra)
        for i in range(size-1):
            return_ls.append(input_ls[slice_ls[i]*i:slice_ls[i]*(i+1)])
        return_ls.append(input_ls[(size-1)*base:])
        return return_ls
    else:
        slice_size = int(input_size/size)
        slice_ls = np.repeat([slice_size],size,axis=0).tolist()
        for i in range(len(slice_ls)):
            return_ls.append(input_ls[slice_ls[i]*i:slice_ls[i]*(i+1)])
        return return_ls
    
    
def extreme_points(lbd,ubd,n):
    r"""Return extreme points (or boundary points) of a list with lower bound 'lbd' and upper bound 'ubd' fixed
    
    The main idea of this function is to partition generate a list based on lower bound and upper bound and then slice that list into list of lists evenly into 'n' parts. Then generate a list of the boundary points for each partitioned part

    Parameters
    ----------
    lbd : float64
        lower bound for the interval (or list)
    ubd : foat64
        upper bound for the interval (or list)
    n : int
        the number of parts the generated list is to be sliced into

    Returns
    -------
    extrema_ls : list, float64
        list of boundary points
    """
    interval = partition_interval(lbd,ubd,n)
    extrema_ls = []
    for i in range(len(interval)):
        extrema_ls.append(interval[i][0])
    return extrema_ls
