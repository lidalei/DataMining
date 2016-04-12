import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier


def plot_surface(ax, clf, X, y):
    h = 0.02  # step size in the mesh
    # create a mesh to plot in
    X_min, X_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(X_min, X_max, h), np.arange(y_min, y_max, h))
    if hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    # Put the result into a color plot 
    z = z.reshape(xx.shape)
    ax.contourf(xx, yy, z, cmap = plt.cm.RdBu_r, alpha = 0.8)
    ax.scatter(X[:, 0], X[:, 1], c = y)
    
if __name__ == '__main__':
    X, y = make_moons(n_samples = 200, random_state = 100)
    
    ns = np.logspace(0, 7, 8, endpoint = True, base = 2.0, dtype = np.int32)
        
    fig, axes = plt.subplots(2, ns.size / 2)
    fig.suptitle('Decision boundaries for Random Forests with different number of trees', fontsize = 'large')
    axes = np.reshape(axes, ns.size)
    
    for n, ax in zip(ns, axes):
        ensemble_clf = RandomForestClassifier(n_estimators = n)
        ensemble_clf.fit(X, y)
        
        ax.set_title('n_estimators = {}'.format(n))
        plot_surface(ax, ensemble_clf, X, y)
    
    plt.tight_layout()
    plt.show()