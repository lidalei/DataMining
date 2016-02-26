# DataMining
Data mining assignments
# NN
This archive contains the data and useful scripts for Set1 homeworks.

It contains the MNIST data set that was downloaded from the e-book by Michael Nielsen: [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html). The dataset description and the dataset in its original formal can be found at [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/).  

The corresponding code for loading the data is located in "**mnist_dataloader.py**".  It allows you load and manipulate data much faster.
# k-means
This archive contains the data and useful scripts for Set1 homeworks.

There are three data sets provided through canvas: C1.txt,C2.txt,C3.txt. 

These data sets all have the following format.  Each line is a data point. The lines have either 3 or 6 tab separated items. The first one is an integer describing the index of the points. The next 2 (or 5 for C3) are the coordinates of the data point. C1 and C2 are in 2 dimensions, and C3 is in 5 dimensions. C1 should have n=20 points, C2 should have n=1004 points, and C3 should have n=1000 points.  We will always measure distance with Euclidean distance

To read the data sets you can use the script "**dataloader_1b.py**".

Python distribution with pre-compiled scientific libraries (SciPy, NumPy, Matplotlib) can be downloaded from the web page: [Anaconda for Windows / OS X / Linux](https://www.continuum.io/downloads)
