import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay


def import_csv(verbose=False):
    csv = pd.read_csv('data_new.csv', header=0, sep=';')
    if verbose: print(csv)
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


def train_test(csv, csv_target, mlp, verbose=False, plot=True):
    sc = PowerTransformer(method='yeo-johnson')
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

    if mlp:
        clf = MLPClassifier(max_iter=1000)  # .fit(X_train, y_train)
        clf.fit(X_train, y_train)
        # For hyperparameter tuning, use train sets. Else, use test sets
        # cv_results = cross_val_score(clf, X_test, y_test, cv=5)
        cv_results = cross_val_score(clf, X_train, y_train, cv=5)
        print(f'Scores: {cv_results}\n'
              f'Average score: {np.mean(cv_results)}')
        score = np.mean(cv_results)
    else:
        clf = LogisticRegressionCV(cv=5, multi_class='ovr', scoring='roc_auc', max_iter=1000000)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

    if verbose:
        # Check for overfitting
        # If this score is much higher than the test score or close to 100% accuracy, the model has likely overfitted
        print(f'Score on TRAIN data: {clf.score(X_train, y_train)}')

    print('score: ', score, '\n')

    if plot:
        if not mlp:
            plot_roc(clf, X_test, y_test)
        plot_confusion_matrix(clf, X_test, y_test)

    return score


def plot_regions(csv, csv_target, mlp):
    pca = PCA(n_components=2)
    sc = PowerTransformer(method='yeo-johnson')

    X_train, X_test, y_train, y_test = train_test_split(csv, csv_target, test_size=0.3)

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    X_train2 = pca.fit_transform(X_train)

    y_train_num = []
    for string in y_train:
        if string == 'Malignant':
            y_train_num.append(1)
        else:
            y_train_num.append(0)

    y_test_num = []
    for string in y_test:
        if string == 'Malignant':
            y_test_num.append(1)
        else:
            y_test_num.append(0)

    if mlp:
        clf = MLPClassifier(max_iter=1000).fit(X_train2, y_train_num)
    else:
        clf = LogisticRegressionCV(cv=5, multi_class='ovr', scoring='roc_auc', max_iter=1000000)
        clf.fit(X_train2, y_train_num)

    X_test2 = pca.fit_transform(X_test)
    PCA_score = clf.score(X_test2, y_test_num)
    print(f'PCA score: {PCA_score}')

    ax = plot_decision_regions(X_train2, np.array(y_train_num), clf=clf)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Benign', 'Malignant'])
    plt.xlabel('PCA feature 1')
    plt.ylabel('PCA feature 2')
    plt.title(f'Score (with PCA, displayed in plot): {PCA_score}')
    plt.show()


def plot_roc(clf, X_test, y_test):
    y_score = clf.decision_function(X_test)
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
MLP = False     # Train an MLP if true, else use Logistic regression

csv, csv_target = import_csv(verbose=False)

scores = []
for i in range(0, 1):
    print(f'Evaluating {i}...')
    scores.append(train_test(csv, csv_target, mlp=MLP, verbose=False, plot=True))
avg_score = sum(scores) / len(scores)
print(f'Average score: {avg_score}')

plot_regions(csv, csv_target, avg_score)  # PCA will perform worse but is necessary for plotting
