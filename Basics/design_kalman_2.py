from kf_book.book_plots import plot_filter
from filterpy.stats import plot_covariance_ellipse
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from kf_book.book_plots import plot_measurements
import numpy as np
from numpy.random import randn
from kf_book.book_plots import plot_track


class ConstantVelocityObject(object):
    def __init__(self, x0=0, vel=1., noise_scale=0.06):
        self.x = x0
        self.vel = vel
        self.noise_scale = noise_scale

    def update(self):
        self.vel += randn() * self.noise_scale
        self.x += self.vel
        return (self.x, self.vel)


def sense(x, noise_scale=1.):
    return x[0] + randn()*noise_scale  # x[0] la self.x


np.random.seed(124)
obj = ConstantVelocityObject()

xs, zs = [], []
for i in range(50):
    x = obj.update()
    z = sense(x)
    xs.append(x)
    zs.append(z)

xs = np.asarray(xs)

plot_track(xs[:, 0])
plot_measurements(range(len(zs)), zs)
plt.legend(loc='best')
plt.show()
