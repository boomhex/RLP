import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def smooth(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    x_new = np.linspace(x.min(), x.max(), 100)
    k = 2          # <= here is the key change
    spl = make_interp_spline(x, y, k=k)
    y_new = spl(x_new)
    return x_new, y_new

x = np.array([0, 10, 20, 30, 50, 100])
train_loss = np.array([0.5, 0.25, 0.1, 0.05, 0.01, 0.01])

x_new, y_new = smooth(x, train_loss)

plt.plot(x_new, y_new, label='train_loss (smooth)')
plt.scatter(x, train_loss, color='black', label='original points')
plt.legend()
plt.show()