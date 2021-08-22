import numpy as np
import book_plots

hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])


def scaled_update(hall, belief, z, z_prob):
    scale = z_prob / (1. - z_prob)
    belief[hall == z] *= scale
    return belief / sum(belief)


belief_init = np.array([0.1] * 10)
belief = scaled_update(hallway, belief_init, z=1, z_prob=.75)

print('belief =', belief)
print('sum =', sum(belief))
print('probability of door =', belief[0])
print('probability of wall =', belief[2])
book_plots.bar_plot(belief, ylim=(0, .3))