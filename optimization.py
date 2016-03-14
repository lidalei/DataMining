from openml.apiconnector import APIConnector
from scipy.io.arff import loadarff
import os
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial import ConvexHull
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold, StratifiedShuffleSplit
from sklearn.ensemble import BaggingClassifier
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
print('Data-set name: %s'%dataset.name)
print(dataset.description)
data, meta = loadarff(dataset.data_file)
target_attribute = dataset.default_target_attribute
target_attribute_names = meta[target_attribute][1]
X, y, attribute_names = dataset.get_dataset(target = target_attribute, return_attribute_names = True)

sss = StratifiedShuffleSplit(y, 1, test_size = 0.2, random_state = 100)
for train_index, test_index in sss:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

