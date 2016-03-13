import numpy as np
import matplotlib.pylab as plt
from scipy.spatial import ConvexHull
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import itertools
import seaborn

## generate dataset
X, y = make_blobs(1000, 2)

