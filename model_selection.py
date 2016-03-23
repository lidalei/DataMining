import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import zero_one_loss


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


def model_selection(clf, parameter, parameter_range): 
    fig, ax = plt.subplots(1, 1)
    
    ## generate dataset
    n_samples, n_centers = 1000, 10
    X, y = make_blobs(n_samples = n_samples, n_features = 2, centers = n_centers, random_state = 100)
    y = np.take([True, False], (y < n_centers / 2))
    
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
    ax.plot(parameter_range, misclassify_rate[:, 0], 'o-', label = '10-fold validation mean')
    # ax.plot(parameter_range, misclassify_rate[:, 1], label = '10-fold validation std')
    # ax.fill_between(parameter_range, misclassify_rate[:, 0] - misclassify_rate[:, 1],
                    # misclassify_rate[:, 0] + misclassify_rate[:, 1], alpha = 0.2, color = "r")

    misclassify_rate_bt, biases, variences = [], [], []
    ## bootstraping
    for k in parameter_range:
        para = {parameter: k}
        clf.set_params(**para)
        scores = []
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
            
            clf.fit(X[train_indices], y[train_indices])
            
            scores.append(zero_one_loss(y[test_indices], clf.predict(X[test_indices])))
            
            preds = clf.predict(X)
            for index in test_indices:
                counts[index] += 1
                sum_preds[index] += preds[index]
        
        test_mask = (counts > 0) # indices of samples that have been tested
        bias, var = bias_var(y[test_mask], sum_preds[test_mask], counts[test_mask])
        biases.append(bias)
        variences.append(var)
        
        misclassify_rate_bt.append([np.mean(scores), np.std(scores)])
    
    misclassify_rate = np.array(misclassify_rate_bt, dtype = np.float64)
    ax.plot(parameter_range, misclassify_rate[:, 0], 'o-', label = '100 boostraping mean')
    # ax.fill_between(parameter_range, misclassify_rate[:, 0] - misclassify_rate[:, 1],
                # misclassify_rate[:, 0] + misclassify_rate[:, 1], alpha=0.2, color="g")
    # ax.plot(parameter_range, misclassify_rate[:, 1], label = '100 boostraping std')
    ax.plot(parameter_range, biases, 'o-', label = '100 boostraping bias')
    ax.plot(parameter_range, variences, 'o-', label = '100 boostraping variance')
    
    ax.grid(True)
    ax.set_ylim(0.0)
    ax.set_xlabel(parameter, fontsize = 'medium')
    ax.set_ylabel('Misclassification rate', fontsize = 'medium')
    ax.legend(loc = 'best', fontsize = 'medium')
    
    plt.show()

if __name__ == '__main__':
    model_selection(KNeighborsClassifier(), 'n_neighbors', xrange(1, 50))
    model_selection(DecisionTreeClassifier(), 'max_depth', xrange(1, 10)) # max_leaf_nodes