from bayesian import *
from Histograms import *

# 1D Packages
import seaborn as sns
import bezier
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import make_interp_spline
import statsmodels.api as sm
# 2D Packages
from scipy.interpolate import SmoothBivariateSpline
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
# 2D integrand
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

# Authors: Johannes Wiesel, Erica Zhang
# Version: February 26, 2023

# DESCRIPTION: This package provides tools to visualize the optimal rho measure obtained via bayesian optimization


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
  
   
def get_curves(source_sample, ts_sample, G0, is_a):
    if is_a:
        curve_y = -0.03
    else:
        curve_y = 0.03
    pt_ls = []
    for i in range(len(source_sample)):
        for j in range(len(ts_sample)):
            if (G0[i][j] != 0):
                temp_mid = (ts_sample[j]+source_sample[i])/2
                temp_num_ls = [source_sample[i],ts_sample[j]]
                tempx = [min(temp_num_ls), temp_mid, max(temp_num_ls)]
                tempy = [0,curve_y,0]
                temp_nodes = np.asfortranarray([tempx, tempy])
                pt_ls.append(temp_nodes)
                
    curve_ls = []
    for i in range(len(pt_ls)):
        curve_ls.append(bezier.Curve(pt_ls[i], degree=2))
    
    plt_curve_ls = []
    number_of_point = 100
    s_vals = np.linspace(0.0, 1.0, number_of_point)
    for i in range(len(curve_ls)):
        plt_curve_ls.append(curve_ls[i].evaluate_multi(s_vals))
    
    return plt_curve_ls

# getting the optimal rho histogram and the optimal rho random variables 
# n is the number of rho samples and d is the dimension
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


# input are samples
def plot_1D_transportMap(a,b,rho_rv):
    a_size = len(a)
    b_size = len(b)
    rho_rv_size = len(rho_rv)
    
    # cast rho_rv into np.array
    rho_rv = np.array(rho_rv)
    
    # compute transport matrix
    x1 = np.ones((a_size,)) / a_size
    x2 = np.ones((rho_rv_size,)) / rho_rv_size
    x3 = np.ones((b_size,)) / b_size
    Ma = ot.dist(a.reshape((a_size, 1)), rho_rv.reshape((rho_rv_size, 1)))
    Mb = ot.dist(rho_rv.reshape((rho_rv_size, 1)), b.reshape((b_size, 1)))
    Ga = ot.emd(x1, x2, Ma)
    Gb = ot.emd(x2, x3, Mb)
    
    curve_a = get_curves(a, rho_rv, Ga, True)
    curve_b = get_curves(rho_rv, b, Gb, False)
    
    pl.rcParams["figure.figsize"] = [7.00, 3.50]
    pl.rcParams["figure.autolayout"] = True
    y_value = 0
    y = np.zeros_like(a) + y_value
    y2 = np.zeros_like(b) + y_value
    y3 = np.zeros_like(rho_rv)+y_value
    pl.plot(a, y,'r+', lw = 5)
    pl.plot(b, y2, 'bx', lw = 5)
    pl.plot(rho_rv, y3, 'g2', lw = 5)
    
    # plot a curves
    for i in range(len(curve_a)):
        pl.plot(curve_a[i][0], curve_a[i][1], color = 'r')
    
    # plot b curves
    for i in range(len(curve_b)):
        pl.plot(curve_b[i][0], curve_b[i][1], color = 'b')
        
    abrho = np.concatenate((a, b, rho_rv))
    
    pl.ylim(-0.03, 0.03)
    pl.xlim(abrho.min()-1, abrho.max()+1)
    pl.show()   
    
    
def plot_samples_histogram(a,b,rho_hist, transport_mat = True):
    p = len(rho_hist)
    rho_grid = np.linspace(-1, 1, p)

    sns.set_style('white')
    sns.kdeplot(np.array(a), color = 'r', bw_method=0.5, label = "source a distribution")
    sns.kdeplot(np.array(b), color = 'b', bw_method=0.5, label = "source b distribution")
    pl.plot(rho_grid, rho_hist, color='g', label='optimal rho distribution')

    pl.legend()
    pl.show()
    
    
# plot transport map as a function    
def transport_function(b,rho_rv):
    rho_rv = np.array(rho_rv)
    b_size = len(b)
    rho_rv_size = len(rho_rv)
    
    print("b:", b)
    print()
    
    print("rho", rho_rv)
    print()
    
    
    x2 = np.ones((rho_rv_size,)) / rho_rv_size
    x3 = np.ones((b_size,)) / b_size
    
    Mb = ot.dist(rho_rv.reshape((rho_rv_size, 1)), b.reshape((b_size, 1)))
    Gb = ot.emd(x2, x3, Mb)
    
    print('Gb', Gb)
    print()
    
    ls = []
    for i in range(rho_rv_size):
        ls.append([x * rho_rv[i].tolist() for x in Gb[i].tolist()])
        
    ls = np.array(ls).reshape(rho_rv_size, b_size)
    
    print(ls)
     
    y_ls = []
    for i in range(b_size):
        y_ls.append(ls.T[i].mean())
        
    lowess = sm.nonparametric.lowess(y_ls, b, frac=0.1)
        
    pl.plot(b, y_ls, '+')
    pl.plot(lowess[:, 0], lowess[:, 1])
    pl.show()

    

# return integrated function of the transport map

def int_function(b,rho_rv):
    rho_rv = np.array(rho_rv)
    b_size = len(b)
    rho_rv_size = len(rho_rv)
    
    x2 = np.ones((rho_rv_size,)) / rho_rv_size
    x3 = np.ones((b_size,)) / b_size
    
    Mb = ot.dist(rho_rv.reshape((rho_rv_size, 1)), np.array(b).reshape((b_size, 1)))
    Gb = ot.emd(x2, x3, Mb)
    
    ls = []
    for i in range(rho_rv_size):
        ls.append([x * rho_rv[i].tolist() for x in Gb[i].tolist()])
        
    ls = np.array(ls).reshape(rho_rv_size, b_size)
     
    y_ls = []
    for i in range(b_size):
        y_ls.append(ls.T[i].mean())
        
    # sort b
    myorder = np.argsort(np.array(b)) # get the sorted index order
    b.sort()
    
    # change order of y_ls accordingly
    y_ls = [y_ls[i] for i in myorder]
        
    f = InterpolatedUnivariateSpline(b, y_ls, k=1)
    
    return f
        

# plot the integrated function of the transport map : smoothed version
def plot_int_function_smooth(b,rho_rv): 
    b = list(b)
    f = int_function(b,rho_rv)
    lbd = min(b)-1
    res_ls = []
    for i in range(len(b)):
        temp = f.integral(lbd,b[i])
        res_ls.append(temp)
        
    X_Y_Spline = make_interp_spline(b, res_ls)
 
    # Returns evenly spaced numbers over a specified interval.
    X = np.linspace(min(b), max(b), 500)
    Y = X_Y_Spline(X)
 
    # Plotting the Graph
    pl.plot(X, Y)
    pl.title("Integrated Function")
    pl.xlabel("Source b")
    pl.show()
    
# plot the integrated function of the transport map : lowess version
def plot_int_function_lowess(b,rho_rv): 
    b = list(b)
    f = int_function(b,rho_rv)
    lbd = min(b)-1
    res_ls = []
    for i in range(len(b)):
        temp = f.integral(lbd,b[i])
        res_ls.append(temp)
        
    lowess = sm.nonparametric.lowess(res_ls, b, frac=0.1)
        
    pl.plot(b, res_ls, '+')
    pl.plot(lowess[:, 0], lowess[:, 1])
    pl.title("Integrated Function")
    pl.xlabel("Source b")
    pl.show()



# 2D convex function
def bivariate_int_function(b,rho_rv):
    rho_rv = np.array(rho_rv)
    b_size = len(b)
    rho_rv_size = len(rho_rv)
    
    x2 = np.ones((rho_rv_size,)) / rho_rv_size
    x3 = np.ones((b_size,)) / b_size
    
    Mb = ot.dist(np.array(b).reshape((b_size, 2)), rho_rv.reshape((rho_rv_size, 2)))
    Gb = ot.emd(x3, x2, Mb) # transport matrix
    
    # calculate the average across rho
    
    rho_mean = []
    for i in range(b_size):
        rho_mean.append(np.dot(Gb[i],rho_rv))
        
    rho_mean = np.array(rho_mean)
        
    # each element corresponds to one point from sample b
    z = []
    for i in range(b_size):
        z.append(np.dot(b[i],rho_mean[i]))
        
    x = b.T[0]
    y = b.T[1]
        
    f = SmoothBivariateSpline(x, y, z, kx=2, ky=2)
    
    return f


def plot_int_bivariatefFunction_trisurf(b,rho_rv): 
    x_ls,y_ls = [b.T[0],b.T[1]]
    f = bivariate_int_function(b,rho_rv)
    x_lbd = x_ls.min()
    y_lbd = y_ls.min()
       
    res_ls = []
    for i in range(len(b)):
        # evaluate integral of the spline over area
        temp = f.integral(x_lbd,b[i][0],y_lbd,b[i][1])
        res_ls.append(temp)
    res_ls = np.array(res_ls)
    
    # plot interoploated smooth surface with irregular spaced points
    triang = mtri.Triangulation(x_ls, y_ls)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    

    ax.triplot(triang, c="#D3D3D3", marker='.', markerfacecolor="#DC143C", markeredgecolor="black", markersize=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
    
    
    # color map
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(1,1,1, projection='3d')
    # Creating color map
    my_cmap = plt.get_cmap('hot')

    trisurf = ax.plot_trisurf(x_ls,y_ls,res_ls,cmap=my_cmap,linewidth = 0.2,antialiased = True,
                         edgecolor = 'grey')
    fig.colorbar(trisurf, ax = ax, shrink = 0.5, aspect = 5)
    ax.scatter(x_ls,y_ls,res_ls, marker='.', s=10, c="black", alpha=0.5)
    #ax.view_init(elev=60, azim=-45)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    

# functions for plotting derivatives
def int_2D_function(b,rho_rv):
    rho_rv = np.array(rho_rv)
    b_size = len(b)
    rho_rv_size = len(rho_rv)
    
    x2 = np.ones((rho_rv_size,)) / rho_rv_size
    x3 = np.ones((b_size,)) / b_size
    
    Mb = ot.dist(np.array(b).reshape((b_size, 2)), rho_rv.reshape((rho_rv_size, 2)))
    Gb = ot.emd(x3, x2, Mb) # transport matrix
    
    # calculate the average across rho
    rho_mean = []
    for i in range(b_size):
        temp = [x * rho_rv[i] for x in Gb[i].tolist()]
        rho_mean.append(sum(temp))
        
    rho_mean = np.array(rho_mean)
        
    # each element corresponds to one point from sample b
    ls = []
    for i in range(b_size):
        ls.append(np.dot(b[i],rho_mean[i]))

    return b,np.array(ls)


def plot_2D_integrad(b, rho_samples, method):
    b, z = int_2D_function(b,rho_samples)
    x, y = [b.T[0],b.T[1]]   
    if method == "Nearest":
        interp = NearestNDInterpolator(list(zip(x.ravel(), y.ravel())),z)
    else:
        interp = LinearNDInterpolator(list(zip(x.ravel(), y.ravel())),z)
    X = np.linspace(x.ravel().min(), x.ravel().max())
    Y = np.linspace(y.ravel().min(), y.ravel().max())
    X, Y = np.meshgrid(X, Y)
    Z = interp(X,Y)
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.plot(x, y, "ok", label="input point")
    plt.legend()
    plt.colorbar()
    plt.axis("equal")
    plt.show()
        
        