from openml.apiconnector import APIConnector
from scipy.io.arff import loadarff
import os
import numpy as np
import matplotlib.pylab as plt
from sklearn.base import clone
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV, ParameterSampler
from sklearn.metrics import zero_one_loss, accuracy_score, confusion_matrix

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
    
    return X, y, attribute_names

def random_search_cv(clf, param_distribution, n_iter_search, X_train, y_train):
    '''
    random search with optimization with nested resampling
    @return: random search object
    '''
    rnd_search = RandomizedSearchCV(clf, param_distributions = param_distribution,
                                    n_iter = n_iter_search, cv = 10)
    rnd_search.fit(X_train, y_train)
    
    return rnd_search

def random_search(clf, param_distribution, n_iter_search, X_train, y_train):
    '''
    random search with optimization without nested resampling
    @return: best_estimator, best score
    '''
    param_list = ParameterSampler(param_distribution, n_iter = n_iter_search)
    best_score = 0.0
    opt_clf = None
    for params in param_list:
        clf.set_params(**params)
        clf.fit(X_train, y_train)
        clf_accuracy = accuracy_score(y_train, clf.predict(X_train))
        if clf_accuracy > best_score:
            best_score = clf_accuracy
            opt_clf = clone(clf)
    
    opt_clf.fit(X_train, y_train)
    
    return opt_clf, best_score

if __name__ == '__main__':
    X, y, attribute_names = get_dataset(32)
    sss = StratifiedShuffleSplit(y, 1, test_size = 0.2, random_state = 100)
    for train_indices, test_indices in sss:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # build a classifier
        clf_cart = DecisionTreeClassifier(class_weight = 'balanced')
        # specify parameters and distributions to sample from
        param_distribution = {"max_depth": sp_randint(1, 100),
                              "min_samples_leaf": sp_randint(1, 100),
                              "min_samples_split": sp_randint(1, 100)}
        # runs of randomized search
        n_iter_search = 64

        rnd_search = random_search_cv(clf_cart, param_distribution, n_iter_search, X_train, y_train)
        print('Random search ' + str(n_iter_search) + ' times')
        print('Best estimator: %s'%rnd_search.best_estimator_)
        print('Best score: %s'%rnd_search.best_score_)
        print('Accuracy: %s'%accuracy_score(y_test, rnd_search.predict(X_test)))
        
        best_clf, best_score = random_search(clf_cart, param_distribution, n_iter_search, X_train, y_train)
        print('Random search withour cross validation ' + str(n_iter_search) + ' times')
        print('Best estimator: %s'%best_clf)
        print('Best score: %s'%best_score)
        print('Accuracy: %s'%accuracy_score(y_test, best_clf.predict(X_test)))

        ## compare random search (cv) with grid search
        grid_dens = xrange(2, 10) # number of samples in each dimension
        rnd_search_iter_times = [i ** 3 for i in grid_dens]
        rnd_search_performances = np.zeros((len(grid_dens), 2), dtype = np.float32)
        grid_search_performances = np.zeros((len(grid_dens), 2), dtype = np.float32)
        for index, (grid_den, n_iter_search) in enumerate(zip(grid_dens, rnd_search_iter_times)):
            ## random search
            rnd_search = random_search_cv(clf_cart, param_distribution, n_iter_search, X_train, y_train)
            rnd_search_performances[index, 0] = rnd_search.best_score_
            rnd_search_performances[index, 1] = accuracy_score(y_test, rnd_search.predict(X_test))
            
            ## grid search
            # use a full grid over all parameters
            param_grid = {"max_depth": np.linspace(1, 100, num = grid_den, dtype = np.int32),
                          "min_samples_leaf": np.linspace(1, 100, num = grid_den, dtype = np.int32),
                          "min_samples_split": np.linspace(1, 100, num = grid_den, dtype = np.int32)}
            
            grid_search = GridSearchCV(clf_cart, param_grid = param_grid, cv = 10)
            grid_search.fit(X_train, y_train)
            grid_search_performances[index, 0] = grid_search.best_score_
            grid_search_performances[index, 1] = accuracy_score(y_test, grid_search.predict(X_test))
        
        ## plot comparison between random search and grid search
        fig, ax = plt.subplots(1, 1)
        ax.plot(rnd_search_iter_times, rnd_search_performances[:, 0], 'o-', label = 'Random search training accuracy')
        ax.plot(rnd_search_iter_times, rnd_search_performances[:, 1], 'o-', label = 'Random search test accuracy')
        ax.plot(rnd_search_iter_times, grid_search_performances[:, 0], '+-', label = 'Grid search training accuracy')
        ax.plot(rnd_search_iter_times, grid_search_performances[:, 1], '+-', label = 'Grid search test accuracy')
        
        ax.grid(True)
        ax.set_title('Random search vs. grid search', fontsize = 'large')
        ax.set_xlabel('Number of search', fontsize = 'medium')
        ax.set_ylabel('Accuracy', fontsize = 'medium')
        ax.legend(loc = 'best', fontsize = 'medium')
        
        plt.show()