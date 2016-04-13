from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from openml import tasks, runs
import xmltodict
import numpy as np

def challenge():    
    ## use dev openml to run
    # Download task, run learner, publish results
    task = tasks.get_task(14951)
    
    ## clf = BaggingClassifier(SVC(), n_estimators = 128)
    
    '''
    clf = RandomForestClassifier(n_estimators = 128, class_weight = 'balanced_subsample')
    '''
    '''
    clf = BaggingClassifier(ExtraTreeClassifier(), n_estimators = 20)
    '''
    '''
    param_grid = {'max_depth': np.linspace(1, 15, num = 15, dtype = np.int64),
                  'class_weight': ['balanced', 'balanced_subsample', None],
                  'min_samples_split': np.linspace(1, 15, num = 15, dtype = np.int64),
                  'criterion': ['gini', 'entropy']
                  }
    base_clf = RandomForestClassifier(n_estimators = 20)
    clf = GridSearchCV(base_clf, param_grid = param_grid, scoring = 'roc_auc',
                       cv = 10, pre_dispatch = '2*n_jobs', n_jobs = 4)
    '''
    '''
    ## grid search - gamma and C, grid_den = 20, time needed = 13.36s
    grid_den = 1
    param_grid = {#'C': np.logspace(-5, 5, num = grid_den, base = 2.0),
                  'gamma': np.logspace(-5, 5, num = grid_den, base = 2.0)
                  }
    clf = GridSearchCV(SVC(probability = True), param_grid = param_grid, scoring = 'roc_auc',
                       cv = 10, pre_dispatch = '2*n_jobs', n_jobs = 4)
    '''
    clf = KNeighborsClassifier(n_neighbors = 5, algorithm = 'brute', metric = 'cosine')
    
    run = runs.run_task(task, clf)
    return_code, response = run.publish()
    
    # get the run id for reference
    if(return_code == 200):
        response_dict = xmltodict.parse(response)
        run_id = response_dict['oml:upload_run']['oml:run_id']
        print("Uploaded run with id %s. Check it at www.openml.org/r/%s" % (run_id,run_id))


if __name__ == '__main__':
    challenge()