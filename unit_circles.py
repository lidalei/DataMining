import numpy as np
import matplotlib.pylab as plt

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

M = 1000
x = np.zeros(M + 1, dtype = np.float64)
y_up = np.zeros(M + 1, dtype = np.float64)
y_down = np.zeros(M + 1, dtype = np.float64)
## l1 norm
for i in range(len(x)):
    x[i] = -1 + 2.0 * i / M
    y_up[i] = 1.0 - np.abs(x[i])
    y_down[i] = -y_up[i]
ax.plot(np.append(x, np.fliplr([x])), np.append(y_up, np.fliplr([y_down])), label = "l1")    
## l2 norm
for i in range(len(x)):
    y_up[i] = (1.0 - x[i] ** 2) ** (0.5)
    y_down[i] = -y_up[i]
ax.plot(np.append(x, np.fliplr([x])), np.append(y_up, np.fliplr([y_down])), label = "l2")

## l10 norm
for i in range(len(x)):
    y_up[i] = (1.0 - x[i] ** 10) ** (0.1)
    y_down[i] = -y_up[i]
ax.plot(np.append(x, np.fliplr([x])), np.append(y_up, np.fliplr([y_down])), label = "l10")
## l infinity norm
ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], label = "linf")
## legend
ax.legend(loc = "upper right")
ax.set_ylim(-1.5, 1.5)
ax.set_xlim(-1.5, 1.5)

plt.show()