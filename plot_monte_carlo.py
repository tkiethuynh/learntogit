import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt


def plot_monte_carlo_mean(xs, ys, f, mean_fx, label, plot_colormap=True):
    fxs, fys = f(xs, ys)

    computed_mean_x = np.average(fxs)
    computed_mean_y = np.average(fys)

    ax = plt.subplot(121)
    ax.grid(b=False)

    #plot_bivariate_colormap(xs, ys)

    plt.scatter(xs, ys, marker='.', alpha=0.02, color='k')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

    ax = plt.subplot(122)
    ax.grid(b=False)

    plt.scatter(fxs, fys, marker='.', alpha=0.02, color='k')
    # plt.scatter(mean_fx[0], mean_fx[1],
    #            marker='v', s=300, c='r', label=label)
    plt.scatter(computed_mean_x, computed_mean_y, marker='*',s=120, c='b', label='Mean')

    #plot_bivariate_colormap(fxs, fys)
    ax.set_xlim([-100, 100])
    ax.set_ylim([-10, 200])
    plt.legend(loc='best', scatterpoints=1)
    print('Difference in mean x={:.3f}, y={:.3f}'.format(
        computed_mean_x-mean_fx[0], computed_mean_y-mean_fx[1]))

def f_nonlinear_xy(x, y):
    return np.array([x + y, .1*x**2 + y*y])


mean = (0., 0.)
p = np.array([[32., 15.], [15., 40.]])
# Compute linearized mean
mean_fx = f_nonlinear_xy(*mean)

# generate random points
xs, ys = multivariate_normal(mean=mean, cov=p, size=30000).T
plot_monte_carlo_mean(xs, ys, f_nonlinear_xy, mean_fx, 'Linearized Mean')
plt.show()