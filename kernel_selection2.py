import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition.tests.test_nmf import random_state
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.metrics import zero_one_loss
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, get_scorer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from matplotlib.colors import Normalize
from sklearn import svm

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    

    
def hot(X, y):
    
    
    C_range = np.logspace(-15, 15, 31,base = 2.0)
    gamma_range = np.logspace(-15, 15, 31, base = 2.0)
     
#     param_grid = dict(gamma=gamma_range, C=C_range)
#     cv = StratifiedShuffleSplit(y, n_iter=10, test_size=0.2, random_state=42)
    roc_auc_scorer = get_scorer("roc_auc")
    scores = []
    for C in C_range:
        for gamma in gamma_range:
            auc_scorer = []
            for train, test in KFold(n=len(X), n_folds=10, random_state=42):
                rbf_svc = svm.SVC(C=C, kernel='rbf', gamma=gamma, probability=True)
                X_train, y_train = X[train], y[train]
                X_test, y_test = X[test], y[test]
                rbf_clf = rbf_svc.fit(X_train, y_train)
                auc_scorer.append(roc_auc_scorer(rbf_clf, X_test, y_test))
            scores.append(np.mean(auc_scorer))
#     grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
#     grid.fit(X, y)
#     scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))
    print scores
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=90)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('AUC')
    plt.show()
     
def SVM_graph(X, y):     
    C_2d_range = [3.0517578125e-05, 0.0312, 1, 32768.0]
    gamma_2d_range = [32768.0, 0.0312, 1, 3.0517578125e-05]
    classifiers = []
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            clf = SVC(C=C, gamma=gamma)
            
.fit(X, y)
            classifiers.append((C, gamma, clf))
     
    plt.subplots(figsize=(20,10))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    h = 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    for (k, (C, gamma, clf)) in enumerate(classifiers):
        plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
        plt.subplots_adjust(wspace=0.1, hspace=0.12)
        # evaluate decision function in a grid
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # visualize decision function for these parameters
        plt.title("gamma=2^%d, C=2^%d" % (np.log2(gamma), np.log2(C)),
                  size='medium')
        # visualize parameter's effect on decision function
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.axis('tight')
    plt.show()  
    
if __name__ == "__main__":
    X , y = make_blobs(n_samples = 1000, centers = 10, random_state=123)
    y = np.take([True, False], (y < 5))
#     hot(X, y)  
    SVM_graph(X, y)