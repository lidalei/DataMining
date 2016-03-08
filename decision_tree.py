from openml.apiconnector import APIConnector
import pandas as pd
import matplotlib.pylab as plt
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image, display

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
X, y, attribute_names = dataset.get_dataset(target = dataset.default_target_attribute, return_attribute_names = True)
y_values = np.unique(y)
fig = plt.figure()
# plot the distribution of target attribute
axes = fig.add_subplot(111)
y_values_distribution, bins, patches = axes.hist(y, bins = y_values.size, facecolor = 'green', alpha=0.5)
axes.set_title('Histogram of ' + str(dataset.default_target_attribute) + ' of ' + dataset.name, fontsize = 'large')
axes.set_xlabel('Values', fontsize = 'medium')
axes.set_ylabel('Count', fontsize = 'medium')
# explore features
# TODO
# axes = fig.add_subplot()
## remove the smaller classes to get a binary problem
# print(y_values_distribution)
kept_y_values = []
for y_value, count in zip(y_values, y_values_distribution):
    if count >= np.sort(y_values_distribution)[-2]:
        kept_y_values.append(y_value)
# print(kept_y_values)
select_indices = np.where(np.in1d(y, kept_y_values))
# select_indices = np.where(np.logical_or(y == kept_y_values[0], y == kept_y_values[1]))
binary_X, binary_y = X[select_indices], y[select_indices]

## CART
clf_cart = DecisionTreeClassifier()
clf_cart.fit(binary_X, binary_y)
export_graphviz(clf_cart, out_file = "cart_dot_data.dot", feature_names = attribute_names,
                # class_names = dataset.default_target_attribute,
                filled = True, rounded = True,
                special_characters = True)

## dot file to pdf or png
'''
dot -Tps cart_dot_data.dot -o cart_dot_data.ps
dot -Tpng cart_dot_data.dot -o cart_dot_data.png
dot -Tpdf cart_dot_data.dot -o cart_dot_data.pdf
'''
## study how stable the trees returned by CART
randomized_indices = np.random.permutation(len(binary_X))
randomized_binary_X, randomized_binary_y = binary_X[randomized_indices][:int(len(randomized_indices) * 0.7)], binary_y[randomized_indices][:int(len(randomized_indices) * 0.7)]
clf_rnd_cart = DecisionTreeClassifier()
clf_rnd_cart.fit(randomized_binary_X, randomized_binary_y)
clf_rnd_cart_dot_data = StringIO()
export_graphviz(clf_rnd_cart, out_file = clf_rnd_cart_dot_data,
                feature_names = attribute_names,
                # class_names = dataset.target_names,
                filled=True, rounded=True,
                special_characters=True)
clf_rnd_cart_graph = pydot.graph_from_dot_data(clf_rnd_cart_dot_data.getvalue())
display(Image(clf_rnd_cart_graph.create_png()))

## randomized tree
clf_rnd_tree = ExtraTreeClassifier()
clf_rnd_tree.fit(binary_X, binary_y)
export_graphviz(clf_rnd_tree, out_file = "rnd_tree_dot_data.dot",
                # class_names = dataset.default_target_attribute,
                filled = True, rounded = True,
                special_characters = True)


plt.show()