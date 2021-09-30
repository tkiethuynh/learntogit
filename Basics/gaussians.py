import numpy as np
import matplotlib.pyplot as plt
import book_plots
from filterpy.stats import gaussian
from scipy.stats import norm
from numpy.random import randn
import random
import math


def normalize(p):
    return p/sum(p)


def update(likelihood, prior):
    return normalize(likelihood*prior)


def mean_var(p):
    x = np.arange(len(p))
    mean = np.sum(p*x, dtype=float)
    var = np.sum((x-mean)**2*p)
    return mean, var


prior = normalize(np.array([4, 2, 0, 7, 2, 12, 35, 20, 3, 2]))
likelihood = normalize(np.array([3, 4, 1, 4, 2, 38, 20, 18, 1, 16]))
posterior = update(likelihood, prior)
#book_plots.bar_plot(posterior)

xs = np.arange(0, 10, 0.01)
mean, var = mean_var(posterior)
#book_plots.bar_plot(posterior)

#plt.plot(xs, gaussian(xs, mean, var, normed=False), c='r')
#plt.plot(xs, norm(mean, var).pdf(xs))


def rand_student_t(df, mu=0, std=1):
    x = random.gauss(0, std)
    y = 2.0 * random.gammavariate(0.5*df, 2.0)
    return x / (math.sqrt(y/df))+mu

def sense_t():
    return 10 + rand_student_t(7)*2

zs = [sense_t() for i in range(5000)]
plt.plot(zs, lw=1)

plt.show()
