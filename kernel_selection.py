import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import get_scorer


def plot_svm(ax, clf, X, y):
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
    # generate data
    n_samples, n_features, centers = 1000, 2, 2
    X, y = make_blobs(n_samples, n_features, centers, random_state = 100)
    
    # set canvas
    fig, ax = plt.subplots(1, 1)
    # ax.scatter(X[:, 0], X[:, 1], c = y)
    
    # SVMs
    linear_svm = SVC(kernel = 'linear')
    poly_svm = SVC(kernel = 'poly')
    rbf_svm = SVC(kernel = 'rbf')
    linear_svm.fit(X, y)
    poly_svm.fit(X, y)
    rbf_svm.fit(X, y)
    
    # plot the decision boundary
    plot_svm(ax, rbf_svm, X, y)
    
    # compute performance with cross validation
    linear_svm_scores = cross_val_score(linear_svm, X, y, cv = 10, n_jobs = -1)
    poly_svm_scores = cross_val_score(poly_svm, X, y, cv = 10, n_jobs = -1)
    rbf_svm_scores = cross_val_score(rbf_svm, X, y, cv = 10, n_jobs = -1)
    # compute performance with AUC
    roc_auc_scorer = get_scorer("roc_auc")
    
    linear_svm_AUC = roc_auc_scorer(linear_svm, X, y)
    poly_svm_AUC = roc_auc_scorer(poly_svm, X, y)
    rbf_svm_AUC = roc_auc_scorer(rbf_svm, X, y)
        
    plt.show()