# Authors: Authors: Johannes Wiesel, Erica Zhang
# License: MIT License
# Version: June 30, 2022

# DESCRIPTION: This package returns a list containing the histogram bins and the histogram height for an array of distributions, including customized. All histogram generating functions include a "plot" parameter that is set by default to be false. Users could turn it on to plot the histograms. 

# import dependencies
import numpy as np
import matplotlib.pylab as pl
from scipy import stats
from itertools import product
import math
from sklearn.utils import check_random_state, deprecated
from partition_tools import *


def gauss1D(n,m,s,plot = False):
    r"""gauss1D histogram

    The main idea of this function is to generate 1D gaussian histogram based on sample size, mean, and standard deviation.

    Parameters
    ----------
    n : int
        sample size (or the number of bins)
    m : float64
        mean
    s : positive float64
        standard deviation
    plot : bool
        plot (TRUE) or not plot (FALSE) the histogram graphs 

    Returns
    -------
    return_ls : list, float64
        return_ls[0]: numpy.ndarray, float64
            histogram bin positions
        return_ls[1]: numpy.ndarray, float64
            histogram heights (or sample weights)
    """
    if s <= 0:
        raise ValueError("Standard Deviation must be strictly greater than zero!")
    # arange bin positions
    x = np.arange(n, dtype=np.float64)-n/2 
    # calculate sample weights using pdf
    h = np.exp(-(x - m) ** 2 / (2 * s ** 2))
    hist = h / h.sum()
    if plot:
        pl.plot(x,hist)
        pl.show()
    return_ls = [x,hist]
    return return_ls


def make_multiD_samples_gauss(n, m, sigma, d, random_state=None):
     r"""Return `n` samples drawn from multi-Dimensional gaussian :math:`\mathcal{N}(m, \sigma)`

    The main idea of this function is to generate multi-dimensional gaussian samples based on sample size, mean, and standard deviation.

    Parameters
    ----------
    n : int
        sample size (or the number of bins)
    m : ndarray, shape (d,)
        mean value of the gaussian distribution
    sigma : ndarray, shape (d, d)
        covariance matrix of the gaussian distribution
    d: int
        dimension
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : ndarray, shape (`n`, 2)
        n samples drawn from :math:`\mathcal{N}(m, \sigma)`.
        
    Reference
    ----------
    This code is adapted from 'make_2D_samples_gauss' from package POT
    """
    if s <= 0:
        raise ValueError("Standard Deviation must be strictly greater than zero!")
    generator = check_random_state(random_state)
    if np.isscalar(sigma):
        sigma = np.array([sigma, ])
    if len(sigma) > 1:
        P = scipy.linalg.sqrtm(sigma)
        res = generator.randn(n, d).dot(P) + m
    else:
        res = generator.randn(n, d) * np.sqrt(sigma) + m
    return res


def create_centered_grid(x,n,v):
    r"""Return a grid of size 'n' that is centered around 'x' with incremental size to be 'v'.

    The main idea of this function is to generate an evenly-spaced centered grid for the purpose of creating ideal bin positions for dirichlet distribution

    Parameters
    ----------
    x : float64
        a number that the resulting grid is centered around
    n : int
        grid size (which in the context of histograms, 'n' is the number of bins)
    v : float64
        size of increment
        
    Returns
    -------
    my_ls : list
        a list of size n that is centered around 'x' with incremental size to be 'v'
    """
    left_size = math.floor((n-1)/2)
    right_size = n-1-left_size
    left_ls = []
    for i in range(left_size):
        myval = x-(i+1)*v
        left_ls.append(myval)
    right_ls = []
    for j in range(right_size):
        myval = x+(j+1)*v
        right_ls.append(myval)
    # concatenate list
    left_ls.append(x)
    my_ls = left_ls+right_ls
    return my_ls


def dirac(x,n,v = 1, plot = False): 
    r"""Return histogram for dirac delta distribution

    The main idea of this function is to generate histogram for a dirac delta distribution based on dirac delta argument 'x', sample size 'n', and cutomizable bins grid incremental size 'v'.

    Parameters
    ----------
    x : float64
        dirac delta argument; dirac measure takes 1 at this point and 0 on all others
    n : int
        sample size (or the number of bins)
    v : float64
        size of increment for the bins grid
    plot : bool
        plot (TRUE) or not plot (FALSE) the histogram graphs

    Returns
    -------
    return_ls : list, float64
        return_ls[0]: numpy.ndarray, float64
            histogram bin positions
        return_ls[1]: numpy.ndarray, float64
            histogram heights (or sample weights)
    """
    bins = create_centered_grid(x,n,v)
    # find index of x in bins
    idx = bins.index(x)
    # create a numpy array of zeros
    hist = np.zeros(n)
    # change the zero at idx to be 1
    hist[idx] = 1
    # plot histogram
    if plot == True:
        pl.plot(bins,hist)
        pl.show()
    return_ls = [bins,hist]
    return return_ls


def combined_dirac(x,n,plot = False):
    r"""Return histogram for a linear combination of dirac delta distribution

    The main idea of this function is to generate histogram for a linear combination of dirac delta distribution based on dirac delta argument 'x', sample size 'n'.

    Parameters
    ----------
    x : float64
        dirac delta argument; dirac measure takes 1 at this point and 0 on all others
    n : int
        sample size (or the number of bins)
    plot : bool
        plot (TRUE) or not plot (FALSE) the histogram graphs

    Returns
    -------
    return_ls : list, float64
        return_ls[0]: numpy.ndarray, float64
            histogram bin positions
        return_ls[1]: numpy.ndarray, float64
            histogram heights (or sample weights)
    """
    # find out how many dirac measures there are
    N = len(x)
    # auto-tuning grid incremental size
    v = abs(max(x)-min(x))/N
    # in case that all the dirac measures are the same
    if v == 0:
        return dirac(x[0],n)
    # equally split sample size 'n' to each dirac measure
    distribution = distribute(n,N)
    # write histogram
    hist = []
    for i in range(N):
        hist_ls = dirac_helper(x[i],distribution[i],v)[1]
        hist = hist+hist_ls.tolist()
    hist = np.array(hist)/N
    # write bins
    bins = []
    for i in range(N):
        bins_ls = dirac(x[i],distribution[i],v)[0]
        bins = bins+bins_ls
    # plot histogram
    if plot == True:
        pl.plot(bins,hist)
        pl.show()
    return_ls = [bins,hist]
    return return_ls


def uniform(n,lbd,ubd,cont = True, plot = False):
     r"""Return histogram for a uniform distribution, either continuous or discrete

    The main idea of this function is to generate histogram for a uniform distribution based on sample size 'n', lower bound 'lbd', upper bound 'ubd', and whether the requested uniform distribution is 'cont', or continuous or not

    Parameters
    ----------
    n : int
        sample size (or the number of bins)
    lbd: float64
        lower bound for the uniform distribution
    ubd: float64
        upper bound for the uniform distribution
    cont : bool
        if the uniform distribution is continuous or not
    plot : bool
        plot (TRUE) or not plot (FALSE) the histogram graphs

    Returns
    -------
    return_ls : list, float64
        return_ls[0]: numpy.ndarray, float64
            histogram bin positions
        return_ls[1]: numpy.ndarray, float64
            histogram heights (or sample weights)
    """
    # assume valid inputs
    x = np.arange(-math.ceil(n/2),n-math.ceil(n/2), dtype=np.float64)  # bin positions
    diff = ubd-lbd
    count = 0
    index_ls = []
    for i in range(n):
        if lbd <= x[i] <= math.floor(ubd)+1:
            count += 1
            index_ls.append(i)
    if count == 0:
        raise ValueError("Lower Bound given is greater than nb of bins given. Decrease lower bound OR increase nb of bins.")
    zero_ls_lower = np.repeat(0,index_ls[0]) # fill in lower zeros
    upper_zeros_nb = n-index_ls[len(index_ls)-1]-1
    zero_ls_upper = np.repeat(0,upper_zeros_nb) # fill in upper zeros
    h = np.repeat(1/diff,count) # fill in nonzero parts
    h = np.concatenate((zero_ls_lower,h,zero_ls_upper),axis=0) # concatenate arrays
    hist = h/h.sum()
    if cont:
        if plot == True:
            pl.plot(x,hist)
            pl.show()
    else:
        if plot == True:
            fig, ax = pl.subplots(1, 1)
            ax.plot(x, hist, 'ro', ms=12, mec='r')
            ax.vlines(x, 0, hist, colors='r', lw=4)
            pl.gcf().set_size_inches(8, 6)
            pl.show()
    return_ls = [x,hist]
    return return_ls


def exponential(n,lam, plot = False):
    r"""Return histogram for an exponential distribution
    
    The main idea of this function is to generate histogram for an exponential distribution based on sample size 'n', exponential parameter lambda 'lam'. 

    Parameters
    ----------
    n : int
        sample size (or the number of bins)
    lam: positive float64
        exponential parameter lambda
    plot : bool
        plot (TRUE) or not plot (FALSE) the histogram graphs

    Returns
    -------
    return_ls : list, float64
        return_ls[0]: numpy.ndarray, float64
            histogram bin positions
        return_ls[1]: numpy.ndarray, float64
            histogram heights (or sample weights)
    """
    # assume valid inputs
    x = np.arange(n, dtype=np.float64)  # bin positions
    h = lam*np.exp(-lam*x)
    hist = h / h.sum()
    if plot:
        pl.plot(x,hist)
        pl.show()
    return_ls = [x,hist]
    return return_ls


def gamma(n,alpha,beta, plot = False):
    r"""Return histogram for a gamma distribution
    
    The main idea of this function is to generate histogram for a gamma distribution based on sample size 'n', gamma parameter alpha 'alpha' and beta 'beta'. 

    Parameters
    ----------
    n : int
        sample size (or the number of bins)
    alpha: positive float64
        gamma parameter alpha
    beta: positive float64
        gamma parameter beta    
    plot : bool
        plot (TRUE) or not plot (FALSE) the histogram graphs

    Returns
    -------
    return_ls : list, float64
        return_ls[0]: numpy.ndarray, float64
            histogram bin positions
        return_ls[1]: numpy.ndarray, float64
            histogram heights (or sample weights)
    """
    # assume valid inputs
    x = np.arange(n, dtype=np.float64)  # bin positions
    gamma_func = scipy.special.gamma(alpha)
    h = (x**(alpha-1))/(gamma_func*(beta**alpha))*np.exp(-x/beta)
    hist = h / h.sum()
    pl.plot(x,hist)
    pl.show()
    return_ls = [x,hist]
    return return_ls


def beta_dist(n, alpha, beta, plot = False):
    r"""Return histogram for a beta distribution
    
    The main idea of this function is to generate histogram for a beta distribution based on sample size 'n', beta parameter alpha 'alpha' and beta 'beta'. 

    Parameters
    ----------
    n : int
        sample size (or the number of bins)
    alpha: positive float64
        beta parameter alpha
    beta: positive float64
        beta parameter beta    
    plot : bool
        plot (TRUE) or not plot (FALSE) the histogram graphs

    Returns
    -------
    return_ls : list, float64
        return_ls[0]: numpy.ndarray, float64
            histogram bin positions
        return_ls[1]: numpy.ndarray, float64
            histogram heights (or sample weights)
    """
    # assume valid inputs
    x = np.arange(n, dtype=np.float64)  # bin positions
    alpha_gamma_func = scipy.special.gamma(alpha)
    beta_gamma_func = scipy.special.gamma(beta)
    sum_gamma_func = scipy.special.gamma(alpha+beta)
    gamma_func = (alpha_gamma_func*beta_gamma_func)/sum_gamma_func
    h = (x**(alpha-1))*((1-x)**(beta-1))/gamma_func
    hist = h / h.sum()
    if plot:
        pl.plot(x,hist)
        pl.show()
    return_ls = [x,hist]
    return return_ls



def poisson(n,lam, plot = False):
    r"""Return histogram for a poisson distribution
    
    The main idea of this function is to generate histogram for a beta distribution based on sample size 'n', poisson parameter lambda 'lam'. 

    Parameters
    ----------
    n : int
        sample size (or the number of bins)
    lam: positive float64
        poisson parameter lambda  
    plot : bool
        plot (TRUE) or not plot (FALSE) the histogram graphs

    Returns
    -------
    return_ls : list, float64
        return_ls[0]: numpy.ndarray, float64
            histogram bin positions
        return_ls[1]: numpy.ndarray, float64
            histogram heights (or sample weights)
    """
    x = np.arange(n, dtype=np.float64)  # bin positions
    factorial_ls = []
    for i in range(n):
        factorial_ls.append(math.factorial(int(x[i])))
    h = (lam**x/np.array(factorial_ls))*math.exp(-lam)
    hist = h/h.sum()
    if plot:
        fig, ax = pl.subplots(1, 1)
        ax.plot(x, hist, 'ro', ms=12, mec='r')
        ax.vlines(x, 0, hist, colors='r', lw=4)
        # change size of plot
        pl.gcf().set_size_inches(8, 6)
        pl.show()
        # smoth curve alternative
        pl.plot(x,hist)
        pl.show()
    return_ls = [x,hist]
    return return_ls



def bernoulli(p, plot = False):
    r"""Return histogram for a bernoulli distribution
    
    The main idea of this function is to generate histogram for a beta distribution based on bernoulli parameter p 'p'. The bin size is automatically set to 2 since bernoulli only takes two values, i.e. 1 and 0.

    Parameters
    ----------
    p : positive float64; in interval [0,1]
        probability of the bernoulli variable taking value 1. 
    plot : bool
        plot (TRUE) or not plot (FALSE) the histogram graphs

    Returns
    -------
    return_ls : list, float64
        return_ls[0]: numpy.ndarray, float64
            histogram bin positions
        return_ls[1]: numpy.ndarray, float64
            histogram heights (or sample weights)
    """
    x = np.array([0,1])  # bin positions
    h = p**x*(1-p)**(1-x)
    hist = h/h.sum()
    if plot:
        fig, ax = pl.subplots(1, 1)
        ax.plot(x, hist, 'ro', ms=12, mec='r')
        ax.vlines(x, 0, hist, colors='r', lw=4)
        # change size of plot
        pl.gcf().set_size_inches(12, 10)
        pl.show()
    return_ls = [x,hist]
    return return_ls
    
    
def binomial(n,N,p, plot = False):
    r"""Return histogram for a binomial distribution
    
    The main idea of this function is to generate histogram for a beta distribution based on sample size 'n', binomial parameter n 'N' (or number of trials), and parameter p 'p'.
    
    Parameters
    ----------
    n : int
        sample size (or number of bins)
    N : int
        binomial parameter 'n' (or the number of trials/bernoulli random variables)
    p : positive float64; in interval [0,1]
        probability of the bernoulli variable taking value 1. 
    plot : bool
        plot (TRUE) or not plot (FALSE) the histogram graphs

    Returns
    -------
    return_ls : list, float64
        return_ls[0]: numpy.ndarray, float64
            histogram bin positions
        return_ls[1]: numpy.ndarray, float64
            histogram heights (or sample weights)
    """
    x = np.arange(n)  # bin positions
    comb_ls = []
    for i in range(n):
        comb_ls.append(math.comb(N,int(x[i])))
    h = np.array(comb_ls)*(p**x)*(1-p)**(N-x)
    hist = h/h.sum()
    if plot:
        fig, ax = pl.subplots(1, 1)
        ax.plot(x, hist, 'ro', ms=12, mec='r')
        ax.vlines(x, 0, hist, colors='r', lw=4)
        # change size of plot
        pl.gcf().set_size_inches(8, 6)
        pl.show()
        # smoth curve alternative
        pl.plot(x,hist)
        pl.show()
    return_ls = [x,hist]
    return return_ls


def geometric(n,p, plot = False):
    r"""Return histogram for a geometric distribution
    
    The main idea of this function is to generate histogram for a geometric distribution based on sample size 'n', geometric parameter p 'p'.
    
    Parameters
    ----------
    n : int
        sample size (or number of bins)
    p : positive float64; in interval [0,1]
        probability of the bernoulli variable taking value 1. 
    plot : bool
        plot (TRUE) or not plot (FALSE) the histogram graphs

    Returns
    -------
    return_ls : list, float64
        return_ls[0]: numpy.ndarray, float64
            histogram bin positions
        return_ls[1]: numpy.ndarray, float64
            histogram heights (or sample weights)
    """
    x = np.arange(n)  # bin positions
    # modify x to be within support of geometric distribution (i.e. exclude 0)
    xg = x[1:]
    h = (1-p)**(xg-1)*p
    h = np.concatenate((np.array([0]),h),axis=0)
    hist = h/h.sum()
    if plot:
        fig, ax = pl.subplots(1, 1)
        ax.plot(x, hist, 'ro', ms=12, mec='r')
        ax.vlines(x, 0, hist, colors='r', lw=4)
        # change size of plot
        pl.gcf().set_size_inches(8, 6)
        pl.show()
        # smoth curve alternative
        pl.plot(x,hist)
        pl.show()
    return_ls = [x,hist]
    return return_ls



# below are the helper functions for customized pdf generation

def check_probability(n):
    r"""Prompts user input for a probability and checks if it is valid
    
    The main idea of this function is to aids request_ls in requesting for a valid probability
    
    Parameters
    ----------
    n : int
        number of points in support (or the number of times requesting and checking for an input probability)

    Returns
    -------
    number_list : list, float64
        customized list of valid the probability
    """
    number_list = []
    for i in range(0, n):
        valid = False
        while not valid:
            print("Enter number at index", i, )
            item = float(input())
            if 0 <= item <= 1:
                valid = True
            else:
                print("Probability p should be in interval [0,1]. Try again!")
                print("\n")
                continue
            valid = True              
        number_list.append(item)
    if sum(number_list) != 1:
        print("Your probability should sum up to 1. Try again!")
        print("\n")
        return check_probability(n = n)
    print("\n")
    print("Your list is ", number_list)
    return number_list


def request_ls(n = None,prob_ls = False):  
    r"""Request user input for both the support of a discrete customized probability distribution and the respective probability mass
    
    The main idea of this function is to aid the customized discrete distribution generation function by prompting appropriate user inputs. The user can either choose to input probability mass directly and correctly or let the function auto-scales the probability.
    
    Parameters
    ----------
    n : int
        length of the return list (or the total number of points in support)
    prob_ls : bool
        TRUE if a probability list (i.e. a list of probability mass wrt each point in the support) is requested

    Returns
    -------
    number_ls : list, float64
        customized list of either the user's customized discrete distribution support or the probability mass
    """
    number_list = []
    if n == None:
        n = int(input("Enter total number of elements in your list: "))
        print("\n")
    else:
        n = n
        print("Note that the algorithm auto scales your probability if it is not valid, i.e. it does not sum up to 1.")
        print("\n")
        valid = False
        while not valid:
            choice = input("If you would like to proceed with auto scaling, type 'y'. Otherwise, type 'n': ")
            if choice == 'n':
                valid = True
                print("You have opted out of auto-scaling. The validity of your input probability will be checked.")
                print("\n")
                return check_probability(n)
            elif choice == 'y':
                valid = True
            else:
                print("You must enter either 'y' or 'n'! Try again.")
                print("\n")     
    for i in range(0, n):
        valid = False
        while not valid:
            print("Enter number at index", i, )
            item = float(input())
            if prob_ls:
                if item >= 0:
                    valid = True
                else:
                    print("Please enter a non-negative number for probability p. Try again!")
                    print("\n")
                    continue
            valid = True              
        number_list.append(item)
    # auto scale probability to make it sum up to 1 
    if prob_ls == True:
        print("Your probabiltiy has been rescaled.")
        print("\n")
        unscaled_num_list = np.array(number_list) 
        unscaled_num_list = unscaled_num_list/unscaled_num_list.max()
        div_factor = 1/unscaled_num_list.sum()
        scaled_num_list = div_factor*unscaled_num_list
        print(unscaled_num_list)
        print("Your list is ", scaled_num_list.tolist())
        return scaled_num_list.tolist()
    else:
        print("Your list is ", number_list)
        return number_list
    
    
    
def request_pdf():
    r"""Request user input for the domain of and the pdf function to generate their customized continuous distribution
    
    The main idea of this function is to aid the customized continuous distribution generation function by prompting appropriate user inputs for the domain of distribution and its pdf
    
    Parameters
    ----------
    NONE

    Returns
    -------
    result : list
        result[0]
        a function of a customized pdf
    """
    result = []
    print("Enter your domain:")
    print("\n")
    # first write a simplified version of domain customization 
    lbd = float(input("Please enter the lower bound of your domain: "))
    ubd = float(input("Please enter the upper bound of your domain: "))
    diff = ubd-lbd
    if diff < 0 :
        print("Upper bound should be strictly greater than lower bound. Try again!")
        return request_pdf()
    # store domain in a list
    domain = [lbd,ubd]
    result.append(domain)
    print("\n")
    print("Your domain interval is: ",domain)
    print("\n")
    func_string = str(input("Please enter your function. Use 'x' to represent your random variable: "))
    # optioinal feature: check if the pdf is valid
    # strip empty spaces in string
    func_string = func_string.strip()
    result.append(func_string)
    return result


def customized_distribution(n = None,pdf_func = None, cont_domain = None, support_ls = None, prob_ls = None, cont = False, plot = False): 
    r"""Generate a customized distribution, either continuous or discrete, histogram
    
    The main idea of this function is to generate historgam sampled from a customized distribution, either continuous or discrete, from user's inputs
    
    Parameters
    ----------
    n : int
        number of points in support (or the number of times requesting and checking for an input probability)
    pdf_func : str
        customized pdf function for a customized continuous distribution
    cont_domain : list
        customized domain for the continuous distribution
    support_ls : list
        customized list of support of discrete distributions
    prob_ls : list
        customized list of probability mass wrt the support
    cont : bool
        TRUE if requesting a customized continuous distribution and FALSE if discrete is requested
    plot : bool
        TRUE if plot the resulting histogram and FALSE otherwise

    Returns
    -------
    return_ls : list, float64
        return_ls[0]: numpy.ndarray, float64
            histogram bin positions
        return_ls[1]: numpy.ndarray, float64
            histogram heights (or sample weights)
    """
    if not cont:
        xg = np.array(support_ls)
        custm = stats.rv_discrete(name='custm', values=(support_ls, prob_ls))
        if plot:
            fig, ax = plt.subplots(1, 1)
            ax.plot(xg, custm.pmf(xg), 'ro', ms=12, mec='r')
            ax.vlines(xg, 0, custm.pmf(xg), colors='r', lw=4)
            # change size of plot
            plt.gcf().set_size_inches(12, 10)
            plt.show()
            # smoth curve alternative
            pl.plot(xg,hist)
            pl.show()
        return [support_ls,prob_ls]
    else:
        # screen domain
        lbd = cont_domain[0]
        ubd = cont_domain[1]
        # automatically scale bin positions to enforce lower bound (QEUSTION)
        xg = np.arange(math.floor(lbd),n+math.floor(lbd), dtype=np.float64) 
        if ubd > xg[n-1]:
            x = xg
        else:
            first_index_lower = xg.tolist().index(math.floor(ubd))
            x_main = xg[:first_index_lower+1] # slice array
            x_upper_zero = np.zeros(n-1-first_index_lower)
            x = np.concatenate((x_main,x_upper_zero),axis=0) # concatenate array 
            h = eval(pdf_func)
            hist = h/h.sum()
            if plot:
                fig, ax = pl.subplots(1, 1)
                ax.plot(xg, hist, 'ro', ms=12, mec='r')
                ax.vlines(xg, 0, hist, colors='r', lw=4)
                # change size of plot
                pl.gcf().set_size_inches(8, 6)
                pl.show()
                # smoth curve alternative
                pl.plot(xg,hist)
                pl.show()
            return_ls = [x,hist]
            return return_ls
        
        

# interactive function that generates a histogram when given array of 1D samples
# prints graph of histogram as side-effect

# one input parameter: name = name of distribution, if any
# interactive function that generates a histogram when given array of 1D samples
# prints graph of histogram as side-effect

# one input parameter: name = name of distribution, if any

def generate_histogram(name = 'Cust', plot = False):
    r"""Generate histogram based on prompted user input for the requested distribution
    
    The main idea of this function is to interactively prompt user's request for the distribution, then prompt for the parameters needed by the respective distribution, and then generate historgam sampled from that distribution. Supported distribution and their names: "Gaussian","Uniform","Exponential", "Gamma", "Beta", "Poisson", "Bernoulli", "Binomial", "Geometric",'Cust'
    
    Parameters
    ----------
    name : string
        number of points in support (or the number of times requesting and checking for an input probability)
    pdf_func : str
        customized pdf function for a customized continuous distribution
    cont_domain : list
        customized domain for the continuous distribution
    support_ls : list
        customized list of support of discrete distributions
    prob_ls : list
        customized list of probability mass wrt the support
    cont : bool
        TRUE if requesting a customized continuous distribution and FALSE if discrete is requested
    plot : bool
        TRUE if plot the resulting histogram and FALSE otherwise

    Returns
    -------
    function returns histogram generating functions which in turn returns:
    
    return_ls : list, float64
        return_ls[0]: numpy.ndarray, float64
            histogram bin positions
        return_ls[1]: numpy.ndarray, float64
            histogram heights (or sample weights)
    """
    if name == None:
        raise ValueError("Please input: (1). the name of the desired distribution under parameter'name', if known OR (2). input an array of samples generated from desired distribution under parameter 's'.")
    # create a list of legal names
    legal_names = ["Gaussian","Uniform","Exponential", "Gamma", "Beta", "Poisson", "Bernoulli", "Binomial", "Geometric",'Cust']
    if name != None:
        if name not in legal_names:
            raise ValueError("The name of the distribution requested is not among the legal inputs for this function. Please check the list of legal distribution names in the API and try again! If you are customizing a distribution, type 'Cust'.")
        else:
            # Gaussian Distribution
            if name == "Gaussian":
                print("You have requested a histogram from a Gaussain Distribution.")
                m = float(input("Mean (type: float): "))
                s = float(input("Standard Deviation (type: float): "))
                bins_num = int(input("Number of bins (type: int): "))
                return gauss1D(n=bins_num,m=m,s=s, plot = plot)
            
            # Uniform Distribution (Continous or Discrete)
            elif name == "Uniform":
                print("You have requested a histogram from a Uniform Distribution.")
                unif_type = input("Continuous or Discrete? Type 'C' for Continuous and 'D' for Discrete.")
                valid_input = False
                while not valid_input:
                    if unif_type == "C" or unif_type == "D":
                        valid_input = True
                    else:
                        print("Must input either 'C' OR 'D'. Try again!")
                if unif_type == 'C':
                    cont = True
                    print("You selected Continuous Uniform Distribution.")
                else:
                    cont = False
                    print("You selected Discrete Uniform Distribution.")
                valid_input = False
                while not valid_input:
                    lbd = float(input("Lower Bound Incusive (type: float): "))
                    ubd = float(input("Upper Bound Inclusive (type: float): "))
                    diff = ubd-lbd
                    if diff > 0:
                        valid_input = True
                    else:
                        print("Upper Bound - Lower Bound should be POSITIVE. Please try again.") 
                bins_num = int(input("Number of bins (type: int): "))
                return uniform(n = bins_num, lbd = lbd, ubd = ubd, cont = cont, plot = plot)
            
            # Exponential Distribution
            elif name == "Exponential":
                print("You have requested a histogram from an Exponential Distribution.")
                valid_input = False
                while not valid_input:
                    lam = float(input("POSITIVE Lambda (type: float): "))
                    if lam > 0:
                        valid_input = True
                    else:
                        print("Lambda should be positive. Try again!")           
                bins_num = int(input("Number of bins (type: int): "))
                return exponential(n = bins_num, lam = lam, plot = plot)
            
            # Gamma Distribution
            elif name == "Gamma":
                print("You have requested a histogram from a Gamma Distribution.")
                valid_input = False
                while not valid_input:
                    alpha = float(input("POSITIVE alpha (type: float): "))
                    beta = float(input("POSITIVE beta (type: float): "))
                    if alpha > 0 and beta > 0:
                        valid_input = True
                    else:
                        print("Both alpha and beta should be positive. Try again!")
                bins_num = int(input("Number of bins (type: int): "))
                return gamma(n = bins_num, alpha = alpha, beta = beta, plot = plot)
            
            # Beta Distribution
            elif name == "Beta":
                print("You have requested a histogram from a Beta Distribution.")
                valid_input = False
                while not valid_input:
                    alpha = float(input("POSITIVE alpha (type: float): "))
                    beta = float(input("POSITIVE beta (type: float): "))
                    if alpha > 0 and beta > 0:
                        valid_input = True
                    else:
                        print("Both alpha and beta should be positive. Try again!")
                bins_num = int(input("Number of bins (type: int): "))
                return beta_dist(n = bins_num, alpha = alpha, beta = beta, plot = plot)
            
            # Poisson Distribution (Poisson Continous Approximation?)
            elif name == "Poisson":
                print("You have requested a histogram from anPoisson Distribution.")
                valid_input = False
                while not valid_input:
                    lam = float(input("POSITIVE Lambda (type: float): "))
                    if lam > 0:
                        valid_input = True
                    else:
                        print("Lambda should be positive. Try again!")           
                bins_num = int(input("Number of bins (type: int): "))
                return poisson(n = bins_num, lam = lam, plot = plot)
            
            # Bernouli Distribution
            elif name == "Bernoulli":
                print("You have requested a histogram from a Bernoulli Distribution.")
                valid_input = False
                while not valid_input:
                    p = float(input("0 <= p <= 1 (type: float): "))
                    if 0 <= p <= 1:
                        valid_input = True
                    else:
                        print("p must be in [0,1]. Try again!")           
                return bernoulli(p = p, plot = plot)
            
            # Binomial Distribution
            elif name == "Binomial":
                print("You have requested a histogram from a Binomial Distribution.")
                valid_input = False
                while not valid_input:
                    p = float(input("0 <= p <= 1 (type: float): "))
                    if 0 <= p <= 1:
                        # check 'N'?
                        N = int(input("N (type: int): "))
                        valid_input = True
                    else:
                        print("p must be in [0,1]. Try again!")           
                bins_num = int(input("Number of bins (type: int): "))
                return binomial(n = bins_num, N = N, p = p, plot = plot)
            
            # Geometric Distribution
            elif name == "Geometric":
                print("You have requested a histogram from a Geometric Distribution.")
                valid_input = False
                while not valid_input:
                    p = float(input("0 <= p <= 1 (type: float): "))
                    if 0 <= p <= 1:
                        valid_input = True
                    else:
                        print("p must be in [0,1]. Try again!")           
                bins_num = int(input("Number of bins (type: int): "))
                return geometric(n = bins_num, p = p, plot = plot)
            
            # Customized Distribution (continuous & discrete)
            elif name == "Cust":
                print("You have requested a histogram from a customized distribution.")
                valid_input = False
                while not valid_input:
                    unif_type = input("Continuous or Discrete? Type 'C' for Continuous and 'D' for Discrete.")
                    if unif_type == "C" or unif_type == "D":
                        valid_input = True
                    else:
                        print("Must input either 'C' OR 'D'. Try again!")
                        continue
                    if unif_type == 'C':
                        cont = True
                        print("You selected Continuous Distribution.")
                        print("\n")
                    else:
                        cont = False
                        print("You selected Discrete Distribution.")
                        print("\n")
                
                if not cont:
                    valid_input = False
                    while not valid_input:
                        print("Please enter a list of your support.")
                        print("\n")
                        support_ls = request_ls()
                        if len(support_ls) == 0:
                            print("Your support list size should be > 0. Try again!")
                            continue
                        print("\n")
                        print("Please enter a list of your point-wise probability with respect to each point in your support.")
                        print("\n")
                        prob_ls = request_ls(n=len(support_ls), prob_ls = True)
                        valid_input = True
                    # request input for number of bins
                    print("\n")
                    return customized_distribution(support_ls = support_ls, prob_ls = prob_ls, cont = False, plot = plot)
                else:
                    # request pdf 
                    print("Please enter a valid pdf: ")
                    print("That is, it has to have non-negative range.")
                    request_result = request_pdf()
                    func_domain = request_result[0]
                    func_string = request_result[1]
                    print("\n")
                    bins_num = int(input("Number of bins (type: int): "))
                    return customized_distribution(n = bins_num, pdf_func = func_string, cont_domain = func_domain, cont = True, plot = plot)
