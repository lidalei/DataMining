import numpy as np
import matplotlib.pylab as plt
from scipy.spatial import ConvexHull
from sklearn.metrics import confusion_matrix
import itertools
import seaborn

true_labels = [True for i in range(7)]
true_labels.extend([False for i in range(6)])
true_labels = np.array(true_labels, dtype = np.bool)
# print(true_labels)
pred_A_labels = np.array([True, True, False, False,
                          True, True, False, False,
                          True, False, False, False, False], dtype = np.bool)
pred_B_labels = np.array([True, True, True, True,
                          False, True, True, False,
                          True, False, True, False, False], dtype = np.bool)
pred_C = np.array([0.8, 0.9, 0.7, 0.6,
                   0.4, 0.8, 0.4, 0.4,
                   0.6, 0.4, 0.4, 0.4, 0.2], dtype = np.float64)
pred_C_labels = np.array(pred_C.shape, dtype = np.bool)

thresholds = np.array([0.2, 0.5, 0.6, 0.8], dtype = np.float64)

def tpr_fpr(conf_mat):
    '''
    @param conf_mat: binary classification confusion matrix
    '''
    tpr = float(conf_mat[0][0]) / (conf_mat[0][0] + conf_mat[0][1])
    fpr = float(conf_mat[1][0]) / (conf_mat[1][0] + conf_mat[1][1])
    return tpr, fpr

fig, ax = plt.subplots(1, 1)
fig.suptitle('ROC of different classifiers', fontsize = 'x-large')
palette = itertools.cycle(seaborn.color_palette(n_colors = 10))

tpr_fpr_points = []

pred_A_conf_mat = confusion_matrix(true_labels, pred_A_labels, labels = [True, False])
pred_A_tpr, pred_A_fpr = tpr_fpr(pred_A_conf_mat)
print('A', pred_A_tpr, pred_A_fpr)
tpr_fpr_points.append([pred_A_fpr, pred_A_tpr])
ax.scatter(pred_A_fpr, pred_A_tpr, label = 'Prediction A', c = next(palette))
ax.annotate('A', xy = (pred_A_fpr, pred_A_tpr), xytext = (0, 2), textcoords = 'offset points')

pred_B_conf_mat = confusion_matrix(true_labels, pred_B_labels, labels = [True, False])
pred_B_tpr, pred_B_fpr = tpr_fpr(pred_B_conf_mat)
print('B', pred_B_tpr, pred_B_fpr)
tpr_fpr_points.append([pred_B_fpr, pred_B_tpr])
ax.scatter(pred_B_fpr, pred_B_tpr, label = 'Prediction B', c = next(palette))
ax.annotate('B', xy = (pred_B_fpr, pred_B_tpr), xytext = (0, 2), textcoords = 'offset points')

for threshold in thresholds:
    pred_C_labels = (pred_C > threshold)
    pred_C_conf_mat = confusion_matrix(true_labels, pred_C_labels, labels = [True, False])
    pred_C_tpr, pred_C_fpr = tpr_fpr(pred_C_conf_mat)
    print('C' + str(threshold), pred_C_tpr, pred_C_fpr)
    tpr_fpr_points.append([pred_C_fpr, pred_C_tpr])
    ax.scatter(pred_C_fpr, pred_C_tpr, label = 'Prediction C, th = ' + str(threshold), c = next(palette))
    ax.annotate('C-' + str(threshold), xy = (pred_C_fpr, pred_C_tpr), xytext = (0, 2), textcoords = 'offset points')

ax.set_xlabel('False positive rate', fontsize = 'large')
ax.set_ylabel('True positive rate', fontsize = 'large')
ax.set_xlim(-0.01, 1.0)
ax.set_ylim(0.0, 1.0)   
ax.legend(loc = 'lower right', fontsize = 'medium', scatterpoints = 1)

## draw iso-cost lines
for cost in np.nditer(np.arange(0.0, 3.2, 0.4)):
    ax.plot([0.0, 1.0], [1 - 0.4 * cost, 1 + 0.2 * 1.0 - 0.4 * cost], '--')
    ax.annotate(str(cost), xy = (0.5, 1 + 0.2 * 0.5 - 0.4 * cost), xytext = (0, 0), textcoords = 'offset points')

## draw convex hull
tpr_fpr_points = np.array(tpr_fpr_points)
convex_hull = ConvexHull(tpr_fpr_points)
for simplex in convex_hull.simplices:
    ax.plot(tpr_fpr_points[simplex, 0], tpr_fpr_points[simplex, 1])

plt.show()