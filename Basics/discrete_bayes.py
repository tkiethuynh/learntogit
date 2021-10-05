import numpy as np
import kf_book.book_plots
import matplotlib.pyplot as plt

hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])


def lh_hallway(hall, z, z_prob):
    try:
        scale = z_prob/(1. - z_prob)
    except ZeroDivisionError:
        scale = 1e8
    likelihood = np.ones(len(hall))
    likelihood[hall == z] *= scale
    return likelihood


def update(likelihood, prior):
    return likelihood*prior/sum(likelihood*prior)


def predict(pdf, offset, kernel):
    N = len(pdf)
    kN = len(kernel)
    width = int((kN - 1) / 2)

    prior = np.zeros(N)
    for i in range(N):
        for k in range(kN):
            index = (i + (width-k) - offset) % N
            prior[i] += pdf[index] * kernel[k]
    return prior

def discrete_bayes_sim(prior, kernel, measurements, z_prob, hallway):
    posterior = np.array([.1]*10)
    priors, posteriors = [], []
    for i, z in enumerate(measurements):
        prior = predict(posterior, 1, kernel)
        priors.append(prior)

        likelihood = lh_hallway(hallway, z, z_prob)
        posterior = update(likelihood, prior)
        posteriors.append(posterior)
    return priors, posteriors

belief_init = np.array([0.1] * 10)
prior = predict(belief_init, offset=3, kernel=[.1, .8, .1])
likelihood = lh_hallway(hallway, z=1, z_prob=.75)
posterior = update(likelihood, belief_init)

plt.subplot(221)
book_plots.bar_plot(belief_init, title='Before prediction', ylim=(0, .3))
plt.subplot(222)
book_plots.bar_plot(posterior, title='After prediction', ylim=(0, .3))

prior = predict(posterior, offset=3, kernel=[.1, .8, .1])
likelihood = lh_hallway(hallway, z=1, z_prob=.75)
posterior = update(likelihood, prior)

plt.subplot(223)
book_plots.bar_plot(prior, title='Before prediction', ylim=(0, .3))
plt.subplot(224)
book_plots.bar_plot(posterior, title='After prediction', ylim=(0, .3))

plt.show()
