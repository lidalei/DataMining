import os, time
from joblib import Parallel, delayed
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

def bias_var(true_preds, sum_preds, counts, n_replicas):
    '''
    compute bias and variance
    @param true_preds: true labels
    @param sum_preds: array of summation of the predictions of each sample
    @param counts: the times each sample is tested (predicted)
    @return: squared bias, variance
    '''
    sample_bias = np.absolute(true_preds - sum_preds / counts)
    sample_var = sample_bias * (1.0 - sample_bias)
    
    weighted_sample_bias_2 = np.power(sample_bias, 2.0) * (counts / n_replicas)
    weighted_sample_var = sample_var * (counts / n_replicas)
    bias = np.mean(weighted_sample_bias_2)
    var = np.mean(weighted_sample_var)
    
    return bias, var

def clf_bias_var(clf, X, y, n_replicas):
        
    roc_auc_scorer = get_scorer("roc_auc")
    # roc_auc_scorer(clf, X_test, y_test)
    auc_scores = []
    error_scores = []
    counts = np.zeros(X.shape[0], dtype = np.float64)
    sum_preds = np.zeros(X.shape[0], dtype = np.float64)
    for it in xrange(n_replicas):
        # generate train sets and test sets
        train_indices = np.random.randint(X.shape[0], size = X.shape[0])
        # get test sets
        in_train = np.unique(train_indices)
        mask = np.ones(X.shape[0], dtype = np.bool)
        mask[in_train] = False
        test_indices = np.arange(X.shape[0])[mask]
        
        clf.fit(X[train_indices], y[train_indices])
        
        auc_scores.append(roc_auc_scorer(clf, X[test_indices], y[test_indices]))
        error_scores.append(zero_one_loss(y[test_indices], clf.predict(X[test_indices])))
        
        preds = clf.predict(X)
        for index in test_indices:
            counts[index] += 1
            sum_preds[index] += preds[index]
    
    test_mask = (counts > 0) # indices of samples that have been tested
    
    # print('counts mean: {}'.format(np.mean(counts)))
    # print('counts standard derivation: {}'.format(np.std(counts)))
    
    bias, var = bias_var(y[test_mask], sum_preds[test_mask], counts[test_mask], n_replicas)
    
    return auc_scores, error_scores, bias, var

if __name__ == '__main__':
    ## get dataset
    X, y, attribute_names, target_attribute_names = get_dataset(44)
    
    ns = np.logspace(11, 0, num = 12, endpoint = True, base = 2.0, dtype = np.int32)
    
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('OOB error versus cross validation error', fontsize = 'x-large')
    ## OOB scores
    oob_err_rates = []
    for n in ns:
        rnd_forest_clf = RandomForestClassifier(n_estimators = n, bootstrap = True, oob_score = True)
        rnd_forest_clf.fit(X, y)
        oob_err_rates.append(1.0 - rnd_forest_clf.oob_score_)
        # plot_surface(ax, rnd_forest_clf, X, y)
    ax.plot(ns, oob_err_rates, '-o', label = 'OOB error')
    
    ## cross validation scores
    cv_err_rates = []
    for n in ns:
        rnd_forest_clf = RandomForestClassifier(n_estimators = n, bootstrap = True, oob_score = False)
        scores = cross_val_score(rnd_forest_clf, X, y, cv = 10, n_jobs = -1)
        cv_err_rates.append([1.0 - np.mean(scores), np.std(scores)])
        # plot_surface(ax, rnd_forest_clf, X, y)
    cv_err_rates = np.array(cv_err_rates)
    ax.plot(ns, cv_err_rates[:, 0], '-o', label = 'Cross validation error')
    # ax.plot(ns, cv_err_rates[:, 1], label = 'CV error std')
    
    ax.grid(True)
    ax.legend(loc = 'best', fontsize = 'large')
    ax.set_xlabel('Number of trees', fontsize = 'large')
    ax.set_ylabel('Error rate', fontsize = 'large')
    ax.set_xlim(np.min(ns) - 1, np.max(ns) + 4)
    
    
    ## compare a single tree with RandomForest ensemble, using 100 bootstrap
    figure, (ax1, ax2) = plt.subplots(2, 1)
    n_replicas = 200
    
    
    # compute bias and variance for a tree
    cart = DecisionTreeClassifier()
    auc_scores, error_scores, bias, var = clf_bias_var(cart, X, y, n_replicas)
    print('auc mean: {}, std: {}'.format(np.mean(auc_scores), np.std(auc_scores)))
    print('error mean: {}, std: {}'.format(np.mean(error_scores), np.std(error_scores)))
    print('bias: {}, var: {}'.format(bias, var))
    
    # ax1.plot(ns[[0, -1]], [bias, bias], '--', label = 'CART bias')
    # ax1.plot(ns[[0, -1]], [var, var], '--', label = 'CART variance')
    
    
    aucs = []
    err_rates = []
    biases_vars = []
    
    start_time = time.time()
    results = Parallel(n_jobs = 8)(delayed(clf_bias_var)(RandomForestClassifier(n_estimators = n, bootstrap = True, oob_score = False),
                                                         X, y, n_replicas) for n in ns)
    print('Time: {}'.format(time.time() - start_time))
    for auc_scores, error_scores, bias, var in results:
        print('auc mean: {}, std: {}'.format(np.mean(auc_scores), np.std(auc_scores)))
        print('error mean: {}, std: {}'.format(np.mean(error_scores), np.std(error_scores)))
        print('squared bias: {}, var: {}'.format(bias, var))
        aucs.append(np.mean(auc_scores))
        err_rates.append(np.mean(error_scores))
        biases_vars.append([bias, var])
    
    biases_vars = np.array(biases_vars)
    
    ax1.plot(ns, aucs, 'o-', label = 'Random Forest AUC scores')
    ax1.legend(loc = 'best', fontsize = 'medium')
    ax1.set_xlabel('Number of trees', fontsize = 'medium')
    ax1.set_xlim(np.min(ns) - 1, np.max(ns) + 4)
    ax1.grid(True, which = 'both')
    
    ax2.plot(ns, err_rates, 'o-', label = 'Random Forest error rate')
    ax2.plot(ns, biases_vars[:, 0], 'o-', label = 'Random forest squared bias')
    ax2.plot(ns, biases_vars[:, 1], 'o-', label = 'Random forest variance')
    ax2.legend(loc = 'best', fontsize = 'medium')
    ax2.set_xlabel('Number of trees', fontsize = 'medium')
    ax2.set_xlim(np.min(ns) - 1, np.max(ns) + 4)
    ax2.grid(True, which = 'both')
    
    plt.tight_layout()
    plt.show()