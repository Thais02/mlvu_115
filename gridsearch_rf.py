import datetime
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PowerTransformer


def import_csv(path, separator, verbose=False):
    csv = pd.read_csv(path, header=0, sep=separator)
    if verbose:
        print(csv)
    csv_target = csv['diagnosis']

    csv = csv.drop(columns=['diagnosis'])

    return csv, csv_target


def gridsearch():
    forest = RandomForestClassifier()

    parameters = {
        'n_estimators': [10, 20, 50, 100, 200, 500, 1000],
        'max_features': ["sqrt", "log2", 3, 5, 7, 10, None]
    }

    clf = GridSearchCV(forest, parameters, cv=5, n_jobs=-1)
    clf.fit(features, target)
    print('Best parameters found by gridsearch:\n', clf.best_params_)
    print('Best score:\n', clf.best_score_)

    return clf.best_params_


############################
# MAIN CODE
############################
no_of_experiments = 200  # Note: this is a SLOW program
range_start = random.randint(0, 9999)

features, target = import_csv(path='data_new.csv', separator=';')
scaler = PowerTransformer()
start = datetime.datetime.now()
params = []

for i in range(range_start, (range_start + no_of_experiments)):
    print(f'Evaluating {i}...')
    param = str(gridsearch())
    params.append(param)

df = pd.DataFrame(params, columns=['best params'])
print(f'\n{df.value_counts()}')
end = datetime.datetime.now()
duration = (end - start).seconds
print(f'\nTesting took {duration} seconds')
