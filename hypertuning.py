import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import requests
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay


def import_csv(verbose=False):
    csv = pd.read_csv('data_new.csv', header=0, sep=';')
    if verbose:
        print(csv)
    csv_target = []
    for row in csv.itertuples():
        if row[1] == 'M':
            csv_target.append('Malignant')
        elif row[1] == 'B':
            csv_target.append('Benign')

    if verbose:
        print('csv targets\n', csv_target, '\n')
        malignant_count = csv_target.count("Malignant")
        malignant_ratio = malignant_count / len(csv_target)
        benign_count = csv_target.count("Benign")
        benign_ratio = benign_count / len(csv_target)
        print(f'{malignant_count} Malignant ({malignant_ratio}%)')
        print(f'{benign_count} Benign ({benign_ratio}%)\n')

    csv = csv.drop(columns=['diagnosis'])
    csv = csv.filter(
        ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
         'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'])

    return csv, csv_target


def splitting_scaling(csv, csv_target, verbose=False):
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(csv, csv_target, test_size=0.3)
    if verbose:
        print('X_train\n', X_train)
        print('X_test\n', X_test)
        print('y_train\n', y_train)
        print('y_test\n', y_test)

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    if verbose:
        print('X_train scaled\n', X_train)
        print('X_test scaled\n', X_test)

    return X_train, X_test, y_train, y_test


def mlp(verbose=False):
    mlp_c = MLPClassifier(max_iter=10000, verbose=False)
    parameters = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (50,)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }
    clf = GridSearchCV(mlp_c, parameters, n_jobs=-1, cv=5)
    clf.fit(features, target)
    print('Best parameters found by gridsearch:\n', clf.best_params_)
    print('Best score:\n', clf.best_score_)

    if verbose:
        # Check for overfitting
        # If this score is much higher than the test score or close to 100% accuracy, the model has likely overfitted
        print(f'Score on TRAIN data: {clf.score(X_train, y_train)}')

    return clf.best_params_


def plot_roc(clf, X_test, y_test):
    try:
        y_score = clf.decision_function(X_test)
    except:
        y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
    area = roc_auc_score(y_test, y_score)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=area).plot()
    plt.title(f'ROC curve\nArea: {area}')
    plt.show()


def plot_confusion_matrix(clf, X_test, y_test):
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=['Benign', 'Malignant'], cmap='Greens')
    plt.show()


############################
# Main code
############################
features, target = import_csv(verbose=False)

start = datetime.datetime.now()
scores = []
params = []
for i in range(0, 100):
    print(f'Evaluating {i}...')
    param = mlp(verbose=False)
    params.append(str(param))
avg_score = sum(scores) / len(scores)
print(f'Average score: {avg_score}')

df = pd.DataFrame(params, columns=['best params'])
print(f'\n{df.value_counts()}')

end = datetime.datetime.now()
duration = (end - start).seconds
print(f'Testing took {duration} seconds')
# requests.post('https://trigger.macrodroid.com/d005595e-fe9e-43fe-8752-1d9c1e04500d/msg', data={'title': 'MLP finished', 'msg': f'Testing took {duration} seconds'})
