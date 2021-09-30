import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

xs = range(500)
ys = randn(500)*1.0 + 10.0
plt.plot(xs, ys)
plt.show()