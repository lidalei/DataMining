import numpy as np
import matplotlib.pylab as plt
from sklearn.svm import SVC

X = np.array([[3, 4], [2, 2], [4, 4], [1, 4], [2, 1], [4, 3], [4, 1]])
y = np.array(['Red', 'Red', 'Red', 'Red', 'Blue', 'Blue', 'Blue'])

linear_svm = SVC(kernel = 'linear', C = 2 ** 15)
linear_svm.fit(X, y)
## w0 * X_1 + w1 * X_2 + b = 0 <=> X_2 = -w0 / w1 * X_1 - b / w1
w = linear_svm.coef_[0]
print('w: {}'.format(w))
print('Margin: %s'%(1.0 / np.linalg.norm(w)))
b = linear_svm.intercept_
print('b: {}'.format(b))
slope = -w[0] / w[1]
## points in the separating line
xx = np.linspace(np.amin(X[:, 0]), np.amax(X[:, 0]))
yy = slope * xx - b / w[1]
## points in the two gutters
yy_top = yy + 1.0 / w[1]
yy_bottom = yy - 1.0 / w[1]
## canvas
fig, ax = plt.subplots(1, 1)
ax.set_title('Maximal margin classifier')
# draw points
ax.scatter(X[:, 0], X[:, 1], c = y)
# draw separating line
ax.plot(xx, yy, 'r-', label = 'Maximal margin classifier')
# draw gutters
ax.plot(xx, yy_top, 'g--', label = 'Red margin line')
ax.plot(xx, yy_bottom, 'b--', label = 'Blue margin line')
# draw support vectors
ax.scatter(linear_svm.support_vectors_[:, 0], linear_svm.support_vectors_[:, 1],
           s = 100, facecolors = 'none', label = 'Support vectors')

if True:
    ax.scatter([2.], [3.], marker = '^', c = ['Blue'], label = 'Additional observation')
    
# set labels
ax.set_xlabel('X_1')
ax.set_ylabel('X_2')
ax.legend(loc = 'best', fontsize = 'medium', scatterpoints = 1)

plt.show()