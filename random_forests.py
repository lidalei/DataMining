import os, time, json
from openml.apiconnector import APIConnector
from scipy.io.arff import loadarff
import numpy as np
import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import get_scorer, zero_one_loss
from sklearn.tree.tree import DecisionTreeClassifier


def get_dataset(did):
    home_dir = os.path.expanduser("~")
    openml_dir = os.path.join(home_dir, ".openml")
    cache_dir = os.path.join(openml_dir, "cache")
    
    with open(os.path.join(openml_dir, "apikey.txt"), 'r') as fh:
        key = fh.readline().rstrip('\n')
    fh.close()
    
    openml = APIConnector(cache_directory = cache_dir, apikey = key)
    dataset = openml.download_dataset(did)
    # print('Data-set name: %s'%dataset.name)
    # print(dataset.description)
    data, meta = loadarff(dataset.data_file)
    target_attribute = dataset.default_target_attribute
    target_attribute_names = meta[target_attribute][1]
    X, y, attribute_names = dataset.get_dataset(target = target_attribute, return_attribute_names = True)
    
    return X, y, attribute_names, target_attribute_names

def bias_var(true_preds, sum_preds, counts):
    '''
    compute bias and variance
    @param true_preds: true labels
    @param sum_preds: array of summation of the predictions of each sample
    @param counts: the times each sample is tested (predicted)
    @return: bias, variance
    '''
    main_preds = np.round(sum_preds / counts)    
    sample_bias = np.abs(true_preds - main_preds)
    sample_var = np.abs(sum_preds / counts - main_preds)
    
    bias = np.mean(sample_bias)
    u_var = np.sum(sample_var[sample_bias == 0]) / float(true_preds.shape[0])
    b_var = np.sum(sample_var[sample_bias != 0]) / float(true_preds.shape[0])
    var = u_var - b_var
    
    return bias, var

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
    ## get dataset
    X, y, attribute_names, target_attribute_names = get_dataset(44)
    
    ns = np.logspace(0, 7, 8, endpoint = True, base = 2.0, dtype = np.int32)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ## OOB scores
    oob_scores = []
    for n in ns:
        rnd_forest_clf = RandomForestClassifier(n_estimators = n, bootstrap = True, oob_score = True, warm_start = True, random_state = 100)
        rnd_forest_clf.fit(X, y)
        oob_scores.append(1.0 - rnd_forest_clf.oob_score_)
        # plot_surface(ax, rnd_forest_clf, X, y)
    ax1.plot(ns, oob_scores, label = 'OOB error')
    
    ax1.legend(loc = 'best', fontsize = 'medium')
    ax1.set_xlabel('Number of base classifiers')
    
    ## cross validation scores
    cv_scores = []
    for n in ns:
        rnd_forest_clf = RandomForestClassifier(n_estimators = n, bootstrap = False, oob_score = False, warm_start = True, random_state = 100)
        scores = cross_val_score(rnd_forest_clf, X, y, cv = 10, n_jobs = -1)
        cv_scores.append([1.0 - np.mean(scores), np.std(scores)])
        # plot_surface(ax, rnd_forest_clf, X, y)
    cv_scores = np.array(cv_scores)
    ax2.plot(ns, cv_scores[:, 0], label = 'cv error mean')
    ax2.plot(ns, cv_scores[:, 1], label = 'cv error std')
    
    ax2.legend(loc = 'best', fontsize = 'medium')
    ax2.set_xlabel('Number of base classifiers')
    
    ## compare a single tree with RandomForest ensemble, using 100 bootstraping
    roc_auc_scorer = get_scorer("roc_auc")
    # roc_auc_scorer(clf, X_test, y_test)
    
    
    # compute bias and variance for a tree
    auc_scores = []
    error_scores = []
    n_replicas = 100
    counts = np.zeros(X.shape[0], dtype = np.int64)
    sum_preds = np.zeros(X.shape[0], dtype = np.float64)
    for it in xrange(n_replicas):
        # generate train sets and test sets
        train_indices = np.random.randint(X.shape[0], size = X.shape[0])
        # get test sets
        in_train = np.unique(train_indices)
        mask = np.ones(X.shape[0], dtype = np.bool)
        mask[in_train] = False
        test_indices = np.arange(X.shape[0])[mask]
        
        cart = DecisionTreeClassifier()
        
        cart.fit(X[train_indices], y[train_indices])
        
        auc_scores.append(roc_auc_scorer(cart, X[test_indices], y[test_indices]))
        error_scores.append(zero_one_loss(y[test_indices], cart.predict(X[test_indices])))
        
        preds = cart.predict(X)
        for index in test_indices:
            counts[index] += 1
            sum_preds[index] += preds[index]
    
    test_mask = (counts > 0) # indices of samples that have been tested
    bias, var = bias_var(y[test_mask], sum_preds[test_mask], counts[test_mask])
    
    print('auc mean: {}, std: {}'.format(np.mean(auc_scores), np.std(auc_scores)))
    print('error mean: {}, std: {}'.format(np.mean(error_scores), np.std(error_scores)))
    print('bias: {}, var: {}'.format(bias, var))
    '''
    for n in ns:
        rnd_forest_clf = RandomForestClassifier(n_estimators = n, bootstrap = True, oob_score = True, warm_start = True, random_state = 100)
    '''
    plt.show()