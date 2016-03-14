import numpy as np
import matplotlib.pylab as plt
from scipy.spatial import ConvexHull
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import zero_one_loss, accuracy_score, confusion_matrix
import itertools
import seaborn

## generate dataset
n_samples, n_centers = 1000, 10
X, y = make_blobs(n_samples = n_samples, n_features = 2, centers = n_centers, random_state = 100)

y = np.take([True, False], (y < 5))

k_candidates = xrange(1, 50)
misclassify_rate = []
for k in k_candidates:
    scores = []
    for train, test in KFold(n = X.shape[0], n_folds = 10, random_state = 100):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        clf = KNeighborsClassifier(n_neighbors = k)
        clf.fit(X_train, y_train)
        scores.append(zero_one_loss(y_test, clf.predict(X_test)))
    
    misclassify_rate.append(np.mean(scores))

fig, ax = plt.subplots(1, 1)
ax.plot(k_candidates, misclassify_rate)
ax.set_ylim(0.0)
ax.set_xlabel('k')
ax.set_ylabel('Misclassification rate')

## bootstraping
BaggingClassifier(KNeighborsClassifier(), )


plt.show()
