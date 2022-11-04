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

from bayesian import *
from Histograms import *

import seaborn as sns

# Authors: Johannes Wiesel, Erica Zhang
# Version: September 30, 2022

# DESCRIPTION: This package provides tools to visualize the optimal rho measure


# histograms generation functions
def generate_normal_hist(x, m, s):
    h = np.exp(-(x - m) ** 2 / (2 * s ** 2))
    hist = h / h.sum()
    return hist

# This is a function for histogram method and gaussian source distributions
def plot_all_hist_gauss(a_size, a_m, a_s, b_size, b_m, b_s, a_grid = np.arange(100, dtype=np.float64), b_grid = np.arange(100, dtype=np.float64), p=5, lbd = 1, ubd = 101, algo = tpe.suggest, max_eval = 300):
    result = bayesian_optimization(method = "hist", a = gauss(a_size,m=a_m, s=a_s), b = gauss(b_size,m=b_m,s=b_s), a_grid = a_grid, b_grid = b_grid, p=p, lbd=lbd, ubd=ubd, algo=algo, max_eval=max_eval, as_dict = False)
    best_alpha = result[1]
    dir_probability = dirichlet.rvs(best_alpha)
    rho_rv = dir_probability.ravel() # discrete probability of each point of the grid
    rho_grid = np.linspace(-1, 1, p)
    mu_hist = generate_normal_hist(rho_grid, m=a_m, s=a_s)
    nu_hist = generate_normal_hist(rho_grid, m=b_m, s=b_s)
    
    pl.plot(graph_grid, mu_hist, color='r', label='source a distribution')
    pl.plot(graph_grid, nu_hist, color='b', label='source b distribution')
    pl.plot(graph_grid, rho_rv, color='y', label='optimal rho distribution')
    
    # Naming the x-axis, y-axis and the whole graph
    pl.xlabel("x-value")
    pl.ylabel("height")
    pl.title("Example 1.0")
    # Adding legend
    pl.legend()
    # Adjust Size
    pl.rcParams['figure.figsize'] = [12, 10]
    # load the display window
    pl.show()

def plot_all_samples(a,b, p=5, target_size = 100, lbd = 1, ubd = 101, algo = tpe.suggest, max_eval = 300):
    result = bayesian_optimization(method = "samples", a = a, b = b, target_size = target_size, lbd = lbd, ubd=ubd, algo=algo, max_eval = max_eval, as_dict = False)
    best_alpha = result[1]
    dir_probability = dirichlet.rvs(best_alpha)
    rho_rv = dir_probability.ravel() # discrete probability of each point of the grid
    rho_grid = np.linspace(-1, 1, p)
    
    sns.set_style('white')
    sns.kdeplot(np.array(a), bw_method=0.5, label = "source a distribution")
    sns.kdeplot(np.array(b), bw_method=0.5, label = "source b distribution")
    pl.plot(rho_grid, rho_rv, color='y', label='optimal rho distribution')

    pl.legend()
    pl.show()
    
def plot_all_dir(a,b, p=5, target_size = 100, lbd = 1, ubd = 101, algo = tpe.suggest, max_eval = 300):
    result = bayesian_optimization(method = "dir", a = a, b = b, target_size = target_size, lbd = lbd, ubd=ubd, algo=algo, max_eval = max_eval, as_dict = False)
    best_rho = result[1] # sampling of rho
    
    sns.set_style('white')
    sns.kdeplot(np.array(a), bw_method=0.5, label = "source a distribution")
    sns.kdeplot(np.array(b), bw_method=0.5, label = "source b distribution")
    sns.kdeplot(np.array(best_rho), bw_method=0.5, label = "optimal rho distribution")

    pl.legend()
    pl.show()