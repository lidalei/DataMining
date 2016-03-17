import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.utils import resample
from sklearn.metrics import zero_one_loss

def model_selection(clf, parameter, parameter_range): 
    fig, ax = plt.subplots(1, 1)
    
    ## generate dataset
    n_samples, n_centers = 10000, 10
    X, y = make_blobs(n_samples = n_samples, n_features = 2, centers = n_centers, random_state = 100)
    # y = np.take([True, False], (y < 5))
    
    misclassify_rate_k_fold = []
    for k in parameter_range:
        para = {parameter: k}
        clf.set_params(**para)
        scores = []
        kfold = KFold(n = X.shape[0], n_folds = 10)
        for train, test in kfold:
            X_train, y_train = X[train], y[train]
            X_test, y_test = X[test], y[test]
            clf.fit(X_train, y_train)
            scores.append(zero_one_loss(y_test, clf.predict(X_test)))
        misclassify_rate_k_fold.append([np.mean(scores), np.std(scores)])
    
    misclassify_rate = np.array(misclassify_rate_k_fold, dtype = np.float64)
    ax.plot(parameter_range, misclassify_rate[:, 0], label = '10-fold validation mean')
    ax.plot(parameter_range, misclassify_rate[:, 1], label = '10-fold validation std')

    misclassify_rate_bt = []
    ## bootstraping
    for k in parameter_range:
        para = {parameter: k}
        clf.set_params(**para)
        scores = []
        for i in xrange(0, 100):
            X_train, y_train = resample(X, y)
            clf.fit(X_train, y_train)
            scores.append(zero_one_loss(y, clf.predict(X)))
        
        misclassify_rate_bt.append([np.mean(scores), np.std(scores)])
    
    misclassify_rate = np.array(misclassify_rate_bt, dtype = np.float64)
    ax.plot(parameter_range, misclassify_rate[:, 0], label = '100 boostraping mean')
    ax.plot(parameter_range, misclassify_rate[:, 1], label = '100 boostraping std')
    
    ax.set_ylim(0.0)
    ax.set_xlabel('k')
    ax.set_ylabel('Misclassification rate')
    ax.legend(loc = 'best', fontsize = 'medium')
    
    plt.show()

if __name__ == '__main__':
    model_selection(KNeighborsClassifier(), 'n_neighbors', xrange(1, 50))
    model_selection(DecisionTreeClassifier(), 'max_depth', xrange(3, 10)) # max_leaf_nodes