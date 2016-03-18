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
import itertools
import seaborn

home_dir = os.path.expanduser("~")
openml_dir = os.path.join(home_dir, ".openml")
cache_dir = os.path.join(openml_dir, "cache")

with open(os.path.join(openml_dir, "apikey.txt"), 'r') as fh:
    key = fh.readline().rstrip('\n')
fh.close()

## load dataset lists
openml = APIConnector(cache_directory = cache_dir, apikey = key)
# datasets = openml.get_dataset_list()
# data = pd.DataFrame(datasets)

dataset = openml.download_dataset(32)
# print('Data-set name: %s'%dataset.name)
# print(dataset.description)
data, meta = loadarff(dataset.data_file)
target_attribute = dataset.default_target_attribute
target_attribute_names = meta[target_attribute][1]
X, y, attribute_names = dataset.get_dataset(target = target_attribute, return_attribute_names = True)

sss = StratifiedShuffleSplit(y, 1, test_size = 0.2, random_state = 100)
for train_index, test_index in sss:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # build a classifier
    clf_cart = DecisionTreeClassifier()
    ## random search with optimization with nested resampling
    # specify parameters and distributions to sample from
    param_distribution = {"max_depth": [3, None],
                  "max_features": sp_randint(1, len(attribute_names)),
                  "min_samples_split": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "criterion": ["gini", "entropy"]}
    # run randomized search
    n_iter_search = 64
    random_search = RandomizedSearchCV(clf_cart, param_distributions = param_distribution,
                                       n_iter = n_iter_search)
    random_search.fit(X_train, y_train)
    print('Best estimator: %s'%random_search.best_estimator_)
    # print('Grid_scores: %s'%random_search.grid_scores_)
    print('Accuracy: %s'%accuracy_score(y_test, random_search.predict(X_test)))
    
    ## random search with optimization without nested resampling
    clf = DecisionTreeClassifier()
    param_list = ParameterSampler(param_distribution, n_iter = n_iter_search)
    max_accuracy, opt_paras_index = 0.0, 0
    opt_clf = None
    for index, params in enumerate(param_list):
        clf.set_params(**params)
        clf.fit(X_train, y_train)
        clf_accuracy = accuracy_score(y_train, clf.predict(X_train))
        if clf_accuracy > max_accuracy:
            max_accuracy = clf_accuracy
            optimal_paras_index = index
            opt_clf = clone(clf)
    
    print('Best estimator: %s'%opt_clf)
    print('Accuracy: %s'%accuracy_score(y_test, opt_clf.fit(X_train, y_train).predict(X_test)))
    
    ## grid search
    # use a full grid over all parameters
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 6, 10],
                  "min_samples_split": [1, 3, 6, 10],
                  "min_samples_leaf": [1, 3, 6, 10],
                  "criterion": ["gini", "entropy"]}
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid = param_grid)
    grid_search.fit(X_train, y_train)
    print('Best estimator: %s'%grid_search.best_estimator_)
    # print('Grid_scores: %s'%grid_search.grid_scores_)
    print('Accuracy: %s'%accuracy_score(y_test, grid_search.predict(X_test)))