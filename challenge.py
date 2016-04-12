from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from openml import tasks, runs
import xmltodict
import numpy as np

## use dev openml to run
# Download task, run learner, publish results
task = tasks.get_task(14951)

## clf = BaggingClassifier(SVC(), n_estimators = 128)

clf = RandomForestClassifier(n_estimators = 128, class_weight = 'balanced_subsample')

'''
## grid search - gamma and C, grid_den = 20, time needed = 13.36s
grid_den = 100
param_grid = {'C': np.logspace(-10, 10, num = grid_den, base = 2.0),
              'gamma': np.logspace(-10, 10, num = grid_den, base = 2.0)
              }
clf = GridSearchCV(SVC(), param_grid = param_grid, scoring = 'roc_auc',
                           cv = 10, pre_dispatch = '2*n_jobs', n_jobs = 4)
'''

run = runs.run_task(task, clf)
return_code, response = run.publish()

# get the run id for reference
if(return_code == 200):
    response_dict = xmltodict.parse(response)
    run_id = response_dict['oml:upload_run']['oml:run_id']
    print("Uploaded run with id %s. Check it at www.openml.org/r/%s" % (run_id,run_id))