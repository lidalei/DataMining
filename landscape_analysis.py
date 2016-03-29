from openml.apiconnector import APIConnector
from scipy.io.arff import loadarff
import os, time, json
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

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
    ## get data
    X, y, attribute_names, target_attribute_names = get_dataset(59)
    
    ## define classifier - rbf svm
    rbf_svm = SVC(kernel = 'rbf')
    
    ## grid search - gamma only
    # use a full grid over all parameters
    param_grid = {'gamma': np.logspace(-15, 15, num = 5000, base = 2.0)}
    grid_search = GridSearchCV(rbf_svm, param_grid = param_grid, scoring = 'roc_auc',
                               cv = 10, pre_dispatch = '2*n_jobs', n_jobs = -1)
    # re-fit on the whole training data
    grid_search.fit(X, y)
    grid_search_scores = [score[1] for score in grid_search.grid_scores_]
    
    # set canvas
    fig, ax = plt.subplots(1, 1)
    # ax.scatter(X[:, 0], X[:, 1], c = y)
    ax.plot(param_grid['gamma'], grid_search_scores)
    ax.set_title('gamma', fontsize = 'large')
    ax.set_xlabel('gamma', fontsize = 'medium')
    ax.set_ylabel('AUC', fontsize = 'medium')
    '''
    start_time = time.time()
    ## grid search - gamma and C, grid_den = 20, time needed = 13.36s
    grid_den = 1000
    param_grid = {'gamma': np.logspace(-15, 15, num = grid_den, base = 2.0),
                  'C': np.logspace(-15, 15, num = grid_den, base = 2.0)}
    grid_search = GridSearchCV(rbf_svm, param_grid = param_grid, scoring = 'roc_auc',
                               cv = 10, pre_dispatch = '2*n_jobs', n_jobs = 4)
    grid_search.fit(X, y)
    
    print(grid_den, 'time: %s'%(time.time() - start_time))
    
    scores = [score[1] for score in grid_search.grid_scores_]
    scores = np.array(scores).reshape(grid_den, grid_den)
    # save scores
    with open('scores.json', 'w+') as f:
        json.dump(scores.tolist(), f)
    f.close()
    fig, ax = plt.subplots(1, 1)
    ax.set_title('AUC = f(C, gamma)', fontsize = 'large')
    # reset rc params to defaults
    im = ax.imshow(scores, interpolation = 'none',
                   cmap = sns.light_palette("green", as_cmap = True))
    # align colormap with heatmap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(mappable = im, ax = ax, cax = cax, orientation = 'vertical')
    ax.set_xlabel('gamma', fontsize = 'medium')
    ax.set_ylabel('C', fontsize = 'medium')
    
    ax.set_xticks(np.arange(grid_den))
    gamma_s = param_grid['gamma'].tolist()
    # save gamma_s
    with open('gamma_s.json', 'w+') as f:
        json.dump(gamma_s, f)
    f.close()
    gamma_s = ['{:0.5f}'.format(gamma) for gamma in gamma_s]
    ax.set_xticklabels(gamma_s, rotation = 45, fontsize = 'smaller')
    ax.set_yticks(np.arange(grid_den))
    C_s = param_grid['C'].tolist()
    # save C_s
    with open('C_s.json', 'w+') as f:
        json.dump(C_s, f)
    f.close()
    C_s = ['{:0.5f}'.format(C) for C in C_s]
    ax.set_yticklabels(C_s, fontsize = 'smaller')
    ax.grid(False)
    # auto-adjust the layout
    fig.tight_layout()
    
    fig.savefig('AUC.pdf')
    '''
    
    plt.show()