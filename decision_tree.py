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
fig1, axes_bar = plt.subplots(1, 1)
# plot the distribution of target attribute
y_values_counts, bin_edges = np.histogram(y, y_values.size, density = False)
print(y_values_counts)
print(target_attribute_names)
# the x locations for the groups
ind = np.arange(y_values.size)
# the width of the bars
bar_width = 0.35
rects = axes_bar.bar(ind, y_values_counts, bar_width, color = 'green', alpha = 0.8, edgecolor = 'green')
# attach some text labels
for rect in rects:
    height = rect.get_height()
    axes_bar.text(rect.get_x() + rect.get_width() / 2.0, height, '%d' % int(height), ha='center', va='bottom')
axes_bar.set_xlim(-bar_width, len(ind) + bar_width)
axes_bar.set_title('Histogram of ' + str(target_attribute) + ' of ' + dataset.name + ' dataset', fontsize = 'x-large')
axes_bar.set_xticks(ind + bar_width / 2.0)
axes_bar.set_xticklabels(target_attribute_names, fontsize = 'medium')
axes_bar.set_xlabel('Class', fontsize = 'large')
axes_bar.set_ylabel('Count', fontsize = 'large')
# explore features



## remove the smaller classes to get a binary problem
kept_y_values = y_values[np.argsort(y_values_counts)[-2:]]
# print(kept_y_values)

select_indices = np.where(np.in1d(y, kept_y_values))
# select_indices = np.where(np.logical_or(y == kept_y_values[0], y == kept_y_values[1]))
bi_class_X, bi_class_y = X[select_indices], y[select_indices]

## CART
clf_cart = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 5)
clf_cart.fit(bi_class_X, bi_class_y)
export_graphviz(clf_cart, out_file = "cart.dot", feature_names = attribute_names,
                class_names = target_attribute_names,
                filled = True, rounded = True,
                special_characters = True)
print(check_output('dot -Tpdf cart.dot -o cart.pdf', shell = True))

## study how stable the trees returned by CART
for i in xrange(2):    
    randomized_indices = np.random.permutation(len(bi_class_X))
    randomized_bi_class_X, randomized_bi_class_y = bi_class_X[randomized_indices][:int(len(randomized_indices) * 0.7)],\
    bi_class_y[randomized_indices][:int(len(randomized_indices) * 0.7)]
    clf_rnd_cart = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 5)
    clf_rnd_cart.fit(randomized_bi_class_X, randomized_bi_class_y)
    output_file_name = 'rnd_cart_run' + str(i + 1) + '.dot'
    export_graphviz(clf_rnd_cart, out_file = output_file_name,
                    feature_names = attribute_names,
                    class_names = target_attribute_names,
                    filled=True, rounded=True,
                    special_characters=True)
    print(check_output('dot -Tpdf ' + output_file_name + ' -o ' + output_file_name.replace('.dot', '.pdf'), shell = True))

## randomized tree
clf_rnd_tree = ExtraTreeClassifier(max_depth = 3, min_samples_leaf = 5)
clf_rnd_tree.fit(bi_class_X, bi_class_y)
export_graphviz(clf_rnd_tree, out_file = 'rnd_tree.dot',
                feature_names = attribute_names,
                class_names = target_attribute_names,
                filled = True, rounded = True,
                special_characters = True)
print(check_output('dot -Tpdf rnd_tree.dot -o rnd_tree.pdf', shell = True))

plt.show()