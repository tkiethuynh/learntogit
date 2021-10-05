import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from collections import namedtuple
import filterpy.stats as stats

xs = range(500)
ys = randn(500)*1.0 + 10.0
#plt.plot(xs, ys)
# plt.show()

####################

#gaussian = namedtuple('Gaussian', ['mean', 'var'])
#gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])

g1 = gaussian(3.4, 10.1)
g2 = gaussian(mean=4.5, var=0.2**2)
print(g1)
print(g2)

#####################

def predict(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)


def gaussian_multiply(g1, g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean)/(g1.var + g2.var)
    variance = (g1.var*g2.var)/(g1.var + g2.var)
    return gaussian(mean, variance)


def update(prior, likelihood):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior

pos = gaussian(10.0, 0.5**2)
move = gaussian(25.0, 0.7**2)
estimated_pos = update(pos, move)
print(estimated_pos)

xs = np.arange(7, 30, 0.1)

ys = [stats.gaussian(x, pos.mean, pos.var) for x in xs]
plt.plot(xs, ys, label='$\mathcal{N}(10,0.04)$')

ys = [stats.gaussian(x, move.mean, move.var) for x in xs]
plt.plot(xs, ys, label='$\mathcal{N}(15,0.49)$', ls='--')

ys = [stats.gaussian(x, estimated_pos.mean, estimated_pos.var) for x in xs]
plt.plot(xs, ys, label='$\mathcal{N}(25,0.43)$', ls='-.')

plt.legend()
#plt.show()

######################
