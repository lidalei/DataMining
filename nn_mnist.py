import os, time, json
from openml.apiconnector import APIConnector
from scipy.io.arff import loadarff
import numpy as np
import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import get_scorer, zero_one_loss
from sklearn.neural_network import MLPClassifier



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

if __name__ == '__main__':
    ## get dataset - MNIST
    X, y, attribute_names, target_attribute_names = get_dataset(554)
    
    ## 60,000 as training data, 10,000 as test data
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    nn_clf = MLPClassifier(hidden_layer_sizes = (100,), algorithm = 'sgd', max_iter = 1000000, learning_rate = 'constant', learning_rate_init = 0.001)
    
    nn_clf.fit(X_train, y_train)
    
    error_rate = zero_one_loss(y_test, nn_clf.predict(X_test))
    
    print('Error rate: {}'.format(error_rate))