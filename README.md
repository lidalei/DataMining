# DataMining
In this repository, various data mining algorithms are implemented while following the Course, Foundations of Data Mining at Eindhoven University of Technology (TU/e). Besides, hyper-parameter tuning techniques are experimented. The algorithms are as follows.
# Files and descriptions.
|File | Algorithm|
|-----|-------------|
|MPNN.py | Multiple processing nearest-neighbor based on Cosine similarity.|
|MTNN.py | Multiple threads nearest-neighbor based on Cosine similarity.|\
|NN.py | Nearest-neighbor based on Cosine similarity.|
|SGDDataset.py | Provides next_batch method, useful in Neural Network Mini-batch training.|
|ada_learning_rate_nn.py | One hidden layer and one Softmax output layer neural netwok based on Tensorflow.|
|challenge.py | Used to challenge the task 14951 in OpenML.|
|dataloader_1b.py | Used to load files in data1b/. Provided by Course Prof.|
|decision_tree.py | Experimented CART and randomized tree with a set of hyperparameter settings.|
|ensembles.py | Experimented with Random Forests.|
|evaluate_NN.py | Used to evaluate Nearest-neighbor algorithms with different distance functions, i.e., confusion matrix.|
|k_means.py | k-means with different initialization methods, inclu. first k points, uniformly sampled k points, kmeans++, gonzales algorithm.|
|k_medians.py | k-median clustering.|
|kernel_selection.py | Support Vector Machines (SVM) with different kernels, incl. linear, rbf and polynomial kernels.|
|kernel_selection2.py | Experimented parameters of SVM with rbf kernel, namely gamma and C.|
|kernel_selection3.py | Grid search of SVM with rbf kernel, using AUC as metric.|
|landscape_analysis.py | Grid search of SVM with rbf kernel. Plot the AUC = f(gamma, C) heat map.|
|max_margin_classifier.py | A simple example to explain support vectors and maximal margin linear classifier.|
|mnist_dataloader.py | To load the MNIST dataset (data1a/). Provided by Course Prof.|
|model_selection.py | Compute bias and variance using bootstraping of knearest-neighbor (different ks) or decision tree (different max_depth or max_leaf_nodes).|
|nn_mnist.py | Neural Network with sklearn.|
|nn_with_alpha.py | Neural Network with different alphas, i.e., l2-norm penalty implemented with Tensorflow.
|nn_with_learning_rate.py| Neural Network with different learning rates implemented with Tensorflow.|
|nn_with_momentum.py| Neural Network with different momentum implemented with Tensorflow.|
|nn_with_nodes.py | Neural Network with a hidden layer and a softmax output layer implemented in Tensorflow.|
|optimization.py | Experimented with different hyperparameter tuning techniques, incl. random search, grid search (with cross validation).|
|random_forests.py | Demonstrate how Random Forests reduce variance without increasing bias (much) so as to reduce the  classification error.|
|random_projection.py | Implement random projection, to do dimensionality reduction. The result is compared with MPNN.py.|
|roc_curves.py | Demonstrate the convex hull of many classifiers in ROC diagram.|
|tensor_flow_softmax_mnist.py | Softmax regression implemented in Tensorflow. This is used to practice with Tensorflow.|
|unit_circles.py | Demonstrate the unit ``circles`` of different norms, inclu. l1, l2, l10 and l-infinity.|










