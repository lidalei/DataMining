import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn import *
from sklearn.decomposition.tests.test_nmf import random_state
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from sklearn.cross_validation import KFold
from sklearn.metrics import zero_one_loss
from sklearn.metrics.scorer import roc_auc_scorer, get_scorer
from sklearn import svm

if __name__ == "__main__":
    X , y = make_blobs(n_samples = 1000, centers = 10, random_state=123)
    y = np.take([True, False], (y < 5))
    
    C_2d_range = [3.0517578125e-05, 0.0312, 1, 32768.0]
    gamma_2d_range = [32768.0, 0.0312, 1, 3.0517578125e-05]
    
#     lin_svc = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#                   decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
#                   max_iter=-1, probability=False, random_state=None, shrinking=True,
#                   tol=0.001,verbose=False)
    rbf_svc = svm.SVC(C=1.0, kernel='rbf', gamma=0.7)
#     poly_svc = svm.SVC(C=1.0, kernel='poly', degree=3)
    
    #palette = itertools.cycle(seaborn.color_palette(n_colors = 10))
    scores_lin = []
    scores_rbf = []
    scores_poly = []
    lin_roc_auc_scorer = []
    rbf_roc_auc_scorer = []
    poly_roc_auc_scorer = []
    roc_auc_scorer = get_scorer("roc_auc")
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            rbf_svc = svm.SVC(C=C, kernel='rbf', gamma=gamma)
            rbf_roc_auc_scorer = []
            for train, test in KFold(n=len(X), n_folds=10, random_state=42):
                X_train, y_train = X[train], y[train]
                X_test, y_test = X[test], y[test]
#                 lin_clf = lin_svc.fit(X_train, y_train)
                rbf_clf = rbf_svc.fit(X_train, y_train)
#                 poly_clf = poly_svc.fit(X_train, y_train)
#                 scores_lin.append(zero_one_loss((y_test),lin_clf.predict(X_test)))
                scores_rbf.append(zero_one_loss((y_test),rbf_clf.predict(X_test)))
#                 scores_poly.append(zero_one_loss((y_test),poly_clf.predict(X_test)))
#                 lin_roc_auc_scorer.append(roc_auc_scorer(lin_clf, X_test, y_test))
                rbf_roc_auc_scorer.append(roc_auc_scorer(rbf_clf, X_test, y_test))
#                 poly_roc_auc_scorer.append(roc_auc_scorer(poly_clf, X_test, y_test)) 
            print 'C='+str(C)+', G='+ str(gamma)     
            print ("RBF AUC = %f +-%f" %(np.mean(rbf_roc_auc_scorer),np.std(rbf_roc_auc_scorer)))
#     print ("linear error = %f +-%f" %(np.mean(scores_lin), np.std(scores_lin)))
#     print ("RBF error = %f +-%f" %(np.mean(scores_rbf), np.std(scores_rbf)))       
#     print ("Polynomial error = %f +-%f" %(np.mean(scores_poly), np.std(scores_poly)))
#     print ("linear AUC = %f +-%f " %(np.mean(lin_roc_auc_scorer), np.std(lin_roc_auc_scorer)))
#     print ("RBF AUC = %f +-%f" %(np.mean(rbf_roc_auc_scorer),np.std(rbf_roc_auc_scorer)))       
#     print ("Polynomial AUC = %f +-%f" %(np.mean(poly_roc_auc_scorer),np.std(poly_roc_auc_scorer)))
    
#     h = 0.2
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     titles = ['SVC with linear kernal',
#               'SVC with RBF kernal',
#               'SVC with polynomial kernel']
#     
#     plt.subplots(figsize=(20,10))
#     for i, clf in enumerate((lin_svc.fit(X, y), rbf_svc.fit(X, y), poly_svc.fit(X, y))):
#         plt.subplot(2, 2, i+1)
#         plt.subplots_adjust(wspace=0.1, hspace=0.1)
#         Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#         Z = Z.reshape(xx.shape)
#         plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
#         plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired)
#         plt.xlim(xx.min(), xx.max())
#         plt.ylim(yy.min(), yy.max())
#         plt.xticks(())
#         plt.yticks(())
#         plt.title(titles[i])
#         
#     plt.show()