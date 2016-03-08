from openml.apiconnector import APIConnector
from scipy.io.arff import loadarff
import pandas as pd
import matplotlib.pylab as plt
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import fbeta_score, confusion_matrix, roc_curve
from subprocess import check_output

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

dataset = openml.download_dataset(10)
# print('Data-set name: %s'%dataset.name)
# print(dataset.description)
data, meta = loadarff(dataset.data_file)
target_attribute = dataset.default_target_attribute
target_attribute_names = meta[target_attribute][1]
X, y, attribute_names = dataset.get_dataset(target = target_attribute, return_attribute_names = True)
y_values = np.unique(y)
fig1, (axes_hist, axes_stem) = plt.subplots(1, 2)
# plot the distribution of target attribute
y_values_counts, bins, patches = axes_hist.hist(y, bins = y_values.size, align = 'mid', facecolor = 'green', alpha=0.5)
print(y_values_counts)
axes_hist.set_title('Histogram of ' + str(target_attribute) + ' of ' + dataset.name, fontsize = 'large')
axes_hist.set_xlabel('Values', fontsize = 'medium')
axes_hist.set_ylabel('Count', fontsize = 'medium')
axes_stem.stem(y_values, y_values_counts)
axes_stem.set_xlim(y_values[0] - 1, y_values[-1] + 1)
axes_stem.set_title('Histogram of ' + str(target_attribute) + ' of ' + dataset.name, fontsize = 'large')
axes_stem.set_xlabel('Values', fontsize = 'medium')
axes_stem.set_ylabel('Count', fontsize = 'medium')
# explore features
# TODO


## remove the smaller classes to get a binary problem
kept_y_values = y_values[np.argsort(y_values_counts)[-2:]]
# print(kept_y_values)

select_indices = np.where(np.in1d(y, kept_y_values))
# select_indices = np.where(np.logical_or(y == kept_y_values[0], y == kept_y_values[1]))
binary_X, binary_y = X[select_indices], y[select_indices]

## CART
clf_cart = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 5)
clf_cart.fit(binary_X, binary_y)
export_graphviz(clf_cart, out_file = "cart.dot", feature_names = attribute_names,
                class_names = target_attribute_names,
                filled = True, rounded = True,
                special_characters = True)
print(check_output('dot -Tpdf cart.dot -o cart.pdf', shell = True))

## study how stable the trees returned by CART
for i in xrange(2):    
    randomized_indices = np.random.permutation(len(binary_X))
    randomized_binary_X, randomized_binary_y = binary_X[randomized_indices][:int(len(randomized_indices) * 0.7)],\
    binary_y[randomized_indices][:int(len(randomized_indices) * 0.7)]
    clf_rnd_cart = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 5)
    clf_rnd_cart.fit(randomized_binary_X, randomized_binary_y)
    output_file_name = 'rnd_cart_run' + str(i + 1) + '.dot'
    export_graphviz(clf_rnd_cart, out_file = output_file_name,
                    feature_names = attribute_names,
                    class_names = target_attribute_names,
                    filled=True, rounded=True,
                    special_characters=True)
    print(check_output('dot -Tpdf ' + output_file_name + ' -o ' + output_file_name.replace('.dot', '.pdf'), shell = True))

## randomized tree
clf_rnd_tree = ExtraTreeClassifier(max_depth = 3, min_samples_leaf = 5)
clf_rnd_tree.fit(binary_X, binary_y)
export_graphviz(clf_rnd_tree, out_file = 'rnd_tree.dot',
                feature_names = attribute_names,
                class_names = target_attribute_names,
                filled = True, rounded = True,
                special_characters = True)
print(check_output('dot -Tpdf rnd_tree.dot -o rnd_tree.pdf', shell = True))

plt.show()