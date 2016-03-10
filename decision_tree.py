from openml.apiconnector import APIConnector
from scipy.io.arff import loadarff
import pandas as pd
import matplotlib.pylab as plt
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import fbeta_score, confusion_matrix, roc_curve, get_scorer
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
print(y_values)
fig1, axes_bar = plt.subplots(1, 1)
# plot the distribution of target attribute
y_values_counts, bin_edges = np.histogram(y, y_values.size, density = False)
print(y_values_counts, bin_edges)
target_attribute_names = np.array(target_attribute_names)
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
for index, attribute in enumerate(attribute_names):
    print(attribute)
    print(np.histogram2d(X[:, index], y, bins = [np.unique(X[:, index]).size, np.unique(y).size])[0])
#     fig_explore, axes_explore = plt.subplots(1, 1)
#     axes_explore.scatter(X[:, index], y)

## remove the smaller classes to get a binary problem
kept_y_values = np.sort(y_values[np.argsort(y_values_counts)[-2:]])
print(kept_y_values)

select_indices = np.where(np.in1d(y, kept_y_values))
# select_indices = np.where(np.logical_or(y == kept_y_values[0], y == kept_y_values[1]))
bi_class_X, bi_class_y = X[select_indices], y[select_indices] - 1
bi_class_target_attrs = target_attribute_names[kept_y_values]
print(bi_class_target_attrs)

'''
## To evaluate the performance, we get the seventy percent of data as training data
## and the remaining thirty percent as test data
rnd_indices = np.random.permutation(len(bi_class_X))
rnd_training_X, rnd_training_y = bi_class_X[rnd_indices[:int(len(rnd_indices) * 0.7)]], bi_class_y[rnd_indices[:int(len(rnd_indices) * 0.7)]]
rnd_test_X, rnd_test_y = bi_class_X[rnd_indices[int(len(rnd_indices) * 0.7):]], bi_class_y[rnd_indices[int(len(rnd_indices) * 0.7):]]
## CART
clf_cart = DecisionTreeClassifier()
clf_cart.fit(rnd_training_X, rnd_training_y)
export_graphviz(clf_cart, out_file = "cart.dot", feature_names = attribute_names,
                class_names = bi_class_target_attrs,
                filled = True, rounded = True,
                special_characters = True)
print(check_output('dot -Tpdf cart.dot -o cart.pdf', shell = True))
print("Precision = %s"%precision_score(rnd_test_y, clf_cart.predict(rnd_test_X)))
print("Recall = %s"%recall_score(rnd_test_y, clf_cart.predict(rnd_test_X)))
print("F = %s"%fbeta_score(rnd_test_y, clf_cart.predict(rnd_test_X), beta=1))
print("Confusion matrix = %s"%confusion_matrix(rnd_test_y, clf_cart.predict(rnd_test_X)))
roc_auc_scorer = get_scorer("roc_auc")
print("ROC AUC = %s"%roc_auc_scorer(clf_cart, rnd_test_X, rnd_test_y))
fpr, tpr, thresholds = roc_curve(rnd_test_y, clf_cart.predict_proba(rnd_test_X)[:, 1])
fig_cart_roc, axes_cart = plt.subplots(1, 1)
axes_cart.plot(fpr, tpr)
axes_cart.set_title("ROC of CART")
axes_cart.set_xlabel("FPR")
axes_cart.set_ylabel("TPR")
axes_cart.set_ylim(0, 1.1)

## randomized tree
clf_rnd_tree = ExtraTreeClassifier()
clf_rnd_tree.fit(rnd_training_X, rnd_training_y)
export_graphviz(clf_rnd_tree, out_file = 'rnd_tree.dot',
                feature_names = attribute_names,
                class_names = bi_class_target_attrs,
                filled = True, rounded = True,
                special_characters = True)
print(check_output('dot -Tpdf rnd_tree.dot -o rnd_tree.pdf', shell = True))
print("Precision = %s"%precision_score(rnd_test_y, clf_rnd_tree.predict(rnd_test_X)))
print("Recall = %s"%recall_score(rnd_test_y, clf_rnd_tree.predict(rnd_test_X)))
print("F = %s"%fbeta_score(rnd_test_y, clf_rnd_tree.predict(rnd_test_X), beta=1))
print("Confusion matrix = %s"%confusion_matrix(rnd_test_y, clf_rnd_tree.predict(rnd_test_X)))
fpr, tpr, thresholds = roc_curve(rnd_test_y, clf_rnd_tree.predict_proba(rnd_test_X)[:, 1])
fig_rnd_tree_roc, axes_rnd_tree = plt.subplots()
axes_rnd_tree.plot(fpr, tpr)
axes_rnd_tree.set_title("ROC of a randomized tree")
axes_rnd_tree.set_xlabel("FPR")
axes_rnd_tree.set_ylabel("TPR")
axes_rnd_tree.set_ylim(0, 1.1)
roc_auc_scorer = get_scorer("roc_auc")
print("ROC AUC = %s"%roc_auc_scorer(clf_rnd_tree, rnd_test_X, rnd_test_y))
'''

## study how stable the trees returned by CART
for i in xrange(5):    
    rnd_indices = np.random.permutation(len(bi_class_X))
    training_indices = rnd_indices[:int(len(rnd_indices) * 0.7)]
    test_indices = rnd_indices[int(len(rnd_indices) * 0.7):]
    training_X, training_y = bi_class_X[training_indices], bi_class_y[training_indices]
    test_X, test_y = bi_class_X[test_indices], bi_class_y[test_indices]
    clf_rnd_cart = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 5)
    clf_rnd_cart.fit(training_X, training_y)
    output_file_name = 'rnd_cart_run' + str(i + 1) + '.dot'
    export_graphviz(clf_rnd_cart, out_file = output_file_name,
                    feature_names = attribute_names,
                    class_names = bi_class_target_attrs,
                    filled=True, rounded=True,
                    special_characters=True)
    print(check_output('dot -Tpdf ' + output_file_name + ' -o ' + output_file_name.replace('.dot', '.pdf'), shell = True))
    print("Precision = %s"%precision_score(test_y, clf_rnd_cart.predict(test_X)))
    print("Recall = %s"%recall_score(test_y, clf_rnd_cart.predict(test_X)))
    print("F = %s"%fbeta_score(test_y, clf_rnd_cart.predict(test_X), beta=1))
    print("Confusion matrix = %s"%confusion_matrix(test_y, clf_rnd_cart.predict(test_X)))
    fpr, tpr, thresholds = roc_curve(test_y, clf_rnd_cart.predict_proba(test_X)[:, 1])
    fig_rnd_tree_roc, axes_rnd_tree = plt.subplots()
    axes_rnd_tree.plot(fpr, tpr)
    axes_rnd_tree.set_title("ROC of a randomized tree")
    axes_rnd_tree.set_xlabel("FPR")
    axes_rnd_tree.set_ylabel("TPR")
    axes_rnd_tree.set_ylim(0, 1.1)
    roc_auc_scorer = get_scorer("roc_auc")
    print("ROC AUC = %s"%roc_auc_scorer(clf_rnd_cart, test_X, test_y))
    
plt.show()