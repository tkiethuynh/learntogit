import kf_book.book_plots as book_plots
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from collections import namedtuple
import filterpy.stats as stats
import kf_book.kf_internal as kf_internal
from kf_book.kf_internal import DogSimulation

xs = range(500)
ys = randn(500)*1.0 + 10.0
#plt.plot(xs, ys)
# plt.show()

####################

gaussian = namedtuple('Gaussian', ['mean', 'var'])
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
#plt.plot(xs, ys, label='$\mathcal{N}(10,0.04)$')

ys = [stats.gaussian(x, move.mean, move.var) for x in xs]
#plt.plot(xs, ys, label='$\mathcal{N}(15,0.49)$', ls='--')

ys = [stats.gaussian(x, estimated_pos.mean, estimated_pos.var) for x in xs]
#plt.plot(xs, ys, label='$\mathcal{N}(25,0.43)$', ls='-.')

plt.legend()
# plt.show()

######################

np.random.seed(13)

process_var = 2.  # variance in the dog's movement
sensor_var = 4.5  # variance in the sensor

x = gaussian(0., 20.**2)  # dog's position, N(0, 20**2)
velocity = 1
dt = 1.  # time step in seconds
process_model = gaussian(velocity*dt, process_var)  # displacement to add to x
N = 25

# simulate dog and get measurements
dog = DogSimulation(
    x0=x.mean,
    velocity=process_model.mean,
    measurement_var=sensor_var,
    process_var=process_model.var)

# create list of measurements
zs = [dog.move_and_sense() for _ in range(N)]

print('PREDICT\t\t\tUPDATE')
print('     x      var\t\t  z\t    x      var')

# perform Kalman filter on measurement z
xs, priors = np.zeros((N, 2)), np.zeros((N, 2))

for i, z in enumerate(zs):
    prior = predict(x, process_model)
    likelihood = gaussian(z, sensor_var)
    x = update(prior, likelihood)

    priors[i] = prior
    xs[i] = x
    kf_internal.print_gh(prior, x, z)

print()
print('final estimate:        {:10.3f}'.format(x.mean))
print('actual final position: {:10.3f}'.format(dog.x))
print(xs)
print(priors)

'''
book_plots.plot_measurements(zs)
book_plots.plot_filter(xs[:, 0], var=priors[:, 1])
book_plots.plot_predictions(priors[:, 0])
book_plots.show_legend()
kf_internal.print_variance(xs)
plt.show()
'''

#################


def volt(voltage, std):
    return voltage + (randn()*std)

temp_change = 0
voltage_std = 0.13000
process_var = 0.05**2
actual_voltage = 16.3

x = gaussian(25.0, 1000.0)
process_model = gaussian(0.0, process_var)

N = 50
zs=[volt(actual_voltage, voltage_std) for i in range(N)]
ps = []
estimates = []

for z in zs:
    prior = predict(x, process_model)
    x = update(prior, gaussian(z, voltage_std**2))

    #save
    estimates.append(x.mean)
    ps.append(x.var)

#plot
book_plots.plot_measurements(zs)
book_plots.plot_filter(estimates, var=np.array(ps))
book_plots.show_legend()
plt.ylim(16, 17)
book_plots.set_labels(x='step', y='volts')
plt.show()
    
plt.plot(ps)
plt.title('Variance')
print('Variance converges to {:.3f}'.format(ps[-1]))
