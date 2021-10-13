import scipy
from scipy.stats import multivariate_normal
from filterpy.stats import gaussian, multivariate_gaussian
import matplotlib.pyplot as plt
import kf_book.mkf_internal as mkf_internal
from numpy.random import randn
import numpy as np
from kf_book.gaussian_internal import plot_correlated_data

W = [1, 2, 3, 4, 5]
H = [5, 6, 7, 8, 9]
cov = np.cov(H, W, bias=1)
print(cov)

###################

X = np.linspace(1, 10, 100)
Y = -(np.linspace(1, 5, 100) + np.sin(X)*0.2)
# plot_correlated_data(X, Y)
print(np.cov(X, Y))

#######################

X = randn(10000)
Y = randn(10000)
# plot_correlated_data(X, Y)
print(np.cov(X, Y))

######################

mean = [2, 15]
cov = [[10, 0],
       [0, 4]]
#mkf_internal.plot_3d_covariance(mean, cov)


#######################

x = [2.5, 6.5]
mu = [2.5, 6.5]
P = [[8.0, 0.0],
     [0.0, 3.0]]
multivariate_gaussian(x, mu, P)
#mkf_internal.plot_3d_sampled_covariance(mu, P)
print(f'{multivariate_normal(mu, P).pdf(x):.4f}')
#mkf_internal.plot_3_covariances()


#########################

from filterpy.stats import plot_covariance_ellipse
import matplotlib.pyplot as plt

P = [[2,0],[0,6]]
#plot_covariance_ellipse((2, 7), P, fc='g', alpha=0.2, 
#                        std=[1, 2, 3],
#                        title='|2 0|\n|0 6|')
#plt.gca().grid(b=False);

##########################

from kf_book.nonlinear_plots import plot_cov_ellipse_colormap
#plot_cov_ellipse_colormap(cov=[[2, 1.2], [1.2, 1.3]]);

#########################

x = [2, 7]
P = [[2, 1.2], [1.2, 2]]
#plot_covariance_ellipse(x, P, fc='g', alpha=0.2)
#plt.show()

##########################

P = [[4, 3.9], [3.9, 4]]

#plot_covariance_ellipse((5,10), P, ec='k', std=[1, 2, 3])
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.show()

##############################
