# TODO If you're using Pycharm, go to "Run" > "Edit Configurations..." and check "Emulate terminal in output console"
# This is to display the fancy graphics correctly. If you already run from a console, it should *just* work
import random
import time
import traceback
from scipy.stats import sem
from pickle import dump, load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from mlxtend.evaluate import combined_ftest_5x2cv, paired_ttest_5x2cv
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay, classification_report, \
    precision_score, balanced_accuracy_score, recall_score, average_precision_score, accuracy_score
from scipy.stats import ttest_ind
from os import mkdir
from alive_progress import alive_bar


class color:
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bold = '\033[1m'
    end = '\033[0m'


try:
    from notify_thais import notify_thais
except:
    pass


############################
# CLASSIFIERS
############################
def Logistic_Regression(X_train, X_test, y_train, y_test, verbose=False, ax=None, pickle=False):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = LogisticRegression(max_iter=1000000, n_jobs=n_jobs)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if ax:
        plot_decision_regions(X_train, y_train, clf=clf, ax=ax)

    if verbose:
        # Check for overfitting
        # If this score is much higher than the test score or close to 100% accuracy, the model has likely overfitted
        print(f'Score on TRAIN data: {clf.score(X_train, y_train)}')

    if pickle:
        with open(classifiers[0].__name__, 'wb') as file:
            dump(clf, file)

    return score, probs, y_pred


def MLP(X_train, X_test, y_train, y_test, verbose=False, ax=None, pickle=False):
    sc = StandardScaler()  # Not necessary to improve accuracy, just for speed
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = MLPClassifier(max_iter=10000,
                        activation='relu', hidden_layer_sizes=(100,), learning_rate='invscaling', solver='adam')
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if ax:
        plot_decision_regions(X_train, y_train, clf=clf, ax=ax)

    if verbose:
        # Check for overfitting
        # If this score is much higher than the test score or close to 100% accuracy, the model has likely overfitted
        print(f'Score on TRAIN data: {clf.score(X_train, y_train)}')

    if pickle:
        with open(classifiers[1].__name__, 'wb') as file:
            dump(clf, file)

    return score, probs, y_pred


def KNN(X_train, X_test, y_train, y_test, verbose=False, ax=None, pickle=False):
    scaler = MinMaxScaler()  # MinMaxScaler provided optimal scaling
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    k = 5  # optimal k according to grid search
    clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=n_jobs)  # euclidean provided best accuracy
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if ax:
        plot_decision_regions(X_train, y_train, clf=clf, ax=ax)

    if pickle:
        with open(classifiers[2].__name__, 'wb') as file:
            dump(clf, file)

    return score, probs, y_pred


def Random_Forest(X_train, X_test, y_train, y_test, verbose=False, ax=None, pickle=False):
    scaler = PowerTransformer()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if ax:
        clf = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs)
    else:
        clf = RandomForestClassifier(n_estimators=100, max_features=3, n_jobs=n_jobs)

    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if ax:
        plot_decision_regions(X_train, y_train, clf=clf, ax=ax)

    if pickle:
        with open(classifiers[3].__name__, 'wb') as file:
            dump(clf, file)

    return score, probs, y_pred


def Gaussian_Naive_Bayes(X_train, X_test, y_train, y_test, verbose=False, ax=None, pickle=False):
    scaler = PowerTransformer()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if ax:
        plot_decision_regions(X_train, y_train, clf=clf, ax=ax)

    if pickle:
        with open(classifiers[4].__name__, 'wb') as file:
            dump(clf, file)

    return score, probs, y_pred


def Support_Vector_Machine(X_train, X_test, y_train, y_test, verbose=False, ax=None, pickle=False):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # X_train = normalize(X_train)  # , axis=0)
    # X_test = normalize(X_test)  # , axis=0)

    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if ax:
        plot_decision_regions(X_train, y_train, clf=clf, ax=ax)

    if pickle:
        with open(classifiers[5].__name__, 'wb') as file:
            dump(clf, file)

    return score, probs, y_pred


def Dummy_Stratified(X_train, X_test, y_train, y_test, verbose=False, pickle=False):
    # Generates random predictions by respecting the training set class distribution
    clf = DummyClassifier(strategy='stratified')
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    if pickle:
        with open(classifiers[6].__name__, 'wb') as file:
            dump(clf, file)

    return score, probs, y_pred


def Dummy_Uniform(X_train, X_test, y_train, y_test, verbose=False, pickle=False):
    # Generates predictions uniformly at random from the list of unique classes observed in y
    clf = DummyClassifier(strategy='uniform')
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    if pickle:
        with open(classifiers[7].__name__, 'wb') as file:
            dump(clf, file)

    return score, probs, y_pred


############################
# HELPER FUNCTIONS
############################
def import_csv(path, separator, verbose=False):
    csv = pd.read_csv(path, header=0, sep=separator)
    if verbose:
        print(csv)
    csv_target = csv['diagnosis']

    if verbose:
        print('csv targets\n', csv_target, '\n')
        malignant_count = csv_target.value_counts()['M']
        malignant_ratio = malignant_count / len(csv_target)
        benign_count = csv_target.value_counts()['B']
        benign_ratio = benign_count / len(csv_target)
        print(f'{malignant_count} Malignant ({round(malignant_ratio * 100)}%)')
        print(f'{benign_count} Benign ({round(benign_ratio * 100)}%)\n')

    try:
        csv = csv.drop(columns=['diagnosis', 'id'])
    except:
        csv = csv.drop(columns=['diagnosis'])

    csv = csv.dropna(how='all', axis='columns')

    # csv = csv.filter(
    #     ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
    #      'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'])

    return csv, csv_target


def splitting(features, target, state, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=state)
    if verbose:
        print('X_train\n', X_train)
        print('X_test\n', X_test)
        print('y_train\n', y_train)
        print('y_test\n', y_test)

    if apply_pca:
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test


def t_test(classifier1, classifier2):
    # Evaluates based on accuracy scores
    results_two_sided = ttest_ind(scores[classifier1], scores[classifier2], alternative='two-sided',
                                  equal_var=False).pvalue
    results_less = ttest_ind(scores[classifier1], scores[classifier2], alternative='less', equal_var=False).pvalue
    results_greater = ttest_ind(scores[classifier1], scores[classifier2], alternative='greater', equal_var=False).pvalue
    if results_less < pvalue_threshold:
        return results_less, classifier2
    elif results_greater < pvalue_threshold:
        return results_greater, classifier1
    else:
        return results_two_sided, None


def comb_f_test(perform_f: bool, classifier1, classifier2, X, y):
    # Evaluates based on accuracy scores
    if classifier1 == classifier2:
        return 1.0, None
    for classifier in classifiers:
        if classifier.__name__ == classifier1:
            with open(classifier.__name__, 'rb') as file:
                clf1 = load(file)
        if classifier.__name__ == classifier2:
            with open(classifier.__name__, 'rb') as file:
                clf2 = load(file)

    if perform_f:
        f, p = combined_ftest_5x2cv(clf1, clf2, X, y, random_seed=range_start)
    else:
        f, p = paired_ttest_5x2cv(clf1, clf2, X, y, random_seed=range_start)

    score1 = np.mean(scores[classifier1])
    score2 = np.mean(scores[classifier2])

    if score1 > score2:
        if p < pvalue_threshold:
            return p, classifier1
        else:
            return p, None
    elif score1 < score2:
        if p < pvalue_threshold:
            return p, classifier2
        else:
            return p, None
    else:
        return p, None


############################
# PLOTTING
############################
def plot_regions(X_train, X_test, y_train, y_test, save_loc='plots'):
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    y_train_num = []
    for string in y_train:
        if string == 'M':
            y_train_num.append(1)
        else:
            y_train_num.append(0)

    y_test_num = []
    for string in y_test:
        if string == 'M':
            y_test_num.append(1)
        else:
            y_test_num.append(0)

    y_train = y_train_num
    y_test = y_test_num

    fig, axes = plt.subplots(1, 2, squeeze=False, figsize=(8, 4))

    if compare_dummy:  # Dummies do not provide useful output
        classifiers.remove(Dummy_Stratified)
        classifiers.remove(Dummy_Uniform)

    with alive_bar(len(classifiers), title='Plotting decision regions', dual_line=True) as bar:
        for index, ax in enumerate(axes.flatten()):
            # Plot regions with train data
            bar.text = f"-> Plotting {classifiers[index].__name__}..."
            score, _, _ = classifiers[index](X_train, X_test, np.array(y_train).astype(np.int_), y_test, ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, ['Benign', 'Malignant'])
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title(f'{classifiers[index].__name__}')
            bar()
    if save:
        plt.savefig(f'{save_loc}/decision_regions.png', bbox_inches='tight')
        plt.savefig(f'{save_loc}/decision_regions.pdf', bbox_inches='tight')
    plt.show()


def plot_roc(X_train, X_test, y_train, y_test, save_loc='plots', joined=True):
    plots = []
    if joined:
        for classifier in classifiers:
            _, probs, _ = classifier(X_train, X_test, y_train, y_test)
            probs = probs[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs, pos_label='M')
            area = roc_auc_score(y_test, probs)
            plots.append(plt.plot(fpr, tpr, label=f'{classifier.__name__} | Area: {str(area)}'))
        plt.title(f'ROC curves')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        if save:
            plt.savefig(f'{save_loc}/roc_joined.png', bbox_inches='tight')
            plt.savefig(f'{save_loc}/roc_joined.pdf', bbox_inches='tight')
        plt.show()
    # Create separate plots
    if compare_dummy:
        fig, axes = plt.subplots(4, 2, squeeze=False, figsize=(14, 22))
    else:
        fig, axes = plt.subplots(3, 2, squeeze=False, figsize=(14, 17))
    for index, ax in enumerate(axes.flatten()):
        _, probs, _ = classifiers[index](X_train, X_test, y_train, y_test)
        probs = probs[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs, pos_label='M')
        area = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, label=f'Area: {str(area)}', color=plots[index][0].get_color())
        ax.set_title(classifiers[index].__name__)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
    if save:
        plt.savefig(f'{save_loc}/roc.png', bbox_inches='tight')
        plt.savefig(f'{save_loc}/roc.pdf', bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(X_train, X_test, y_train, y_test, save_loc='plots'):
    if compare_dummy:
        fig, axes = plt.subplots(1, 3, squeeze=False, figsize=(21, 5))
    else:
        fig, axes = plt.subplots(3, 2, squeeze=False, figsize=(14, 17))
    for index, ax in enumerate(axes.flatten()):
        _, _, y_pred = classifiers_plot[index](X_train, X_test, y_train, y_test)
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax, display_labels=['Benign', 'Malignant'], cmap='Blues')
        ax.set_title(f'{classifiers_plot[index].__name__}')
    if save:
        plt.savefig(f'{save_loc}/confusionmatrix.png', bbox_inches='tight')
        plt.savefig(f'{save_loc}/confusionmatrix.pdf', bbox_inches='tight')
    plt.show()


def plot_bar(dic, name, y_label, title, save_loc='plots'):
    fig, (ax, lax) = plt.subplots(2, 1)
    min_ = 100
    try:
        del dic[Dummy_Stratified.__name__]
    except:
        pass
    try:
        del dic[Dummy_Uniform.__name__]
    except:
        pass
    print(name)
    for func, l in dic.items():
        l = [x * 100 for x in l]
        err = np.std(l)
        # err = sem(l)
        ax.bar(func, np.mean(l), label=func, yerr=err)
        if min(l) < min_:
            min_ = min(l)
    ax.set_ylabel(f' Mean {y_label}')
    ax.set_ylim(min_ - 0.5, 100)
    ax.set_xticklabels([])
    ax.set_title(f'Mean {name} score for each classifier')
    ax.grid(True, axis='y')

    handles, labels = ax.get_legend_handles_labels()
    lax.legend(handles, labels, borderaxespad=0, loc='upper center')
    lax.axis("off")

    plt.tight_layout()
    if save:
        plt.savefig(f'{save_loc}/bar_{name}.png', bbox_inches='tight')
        plt.savefig(f'{save_loc}/bar_{name}.pdf', bbox_inches='tight')
    plt.show()


def plot_line(dic, name, y_label, title, linestyle='solid', save_loc='plots', joined=True):
    plots = []
    if joined:
        fig, (ax, lax) = plt.subplots(2, 1)
        for func, l in dic.items():
            l = [x * 100 for x in l]
            plots.append(ax.plot(range(0, len(l)), l,
                                 label=func, marker='.', linestyle=linestyle))
        ax.set_ylabel(y_label)
        ax.set_xticks([])
        ax.set_title(title)
        ax.grid(True)

        handles, labels = ax.get_legend_handles_labels()
        lax.legend(handles, labels, borderaxespad=0, loc='upper center')
        lax.axis("off")

        plt.tight_layout()
        if save:
            plt.savefig(f'{save_loc}/line_joined_{name}.png', bbox_inches='tight')
            plt.savefig(f'{save_loc}/line_joined_{name}.pdf', bbox_inches='tight')
        plt.show()
    # Create separate plots
    if compare_dummy:
        fig, axes = plt.subplots(8, 1, squeeze=False, figsize=(14, 17))
    else:
        fig, axes = plt.subplots(6, 1, squeeze=False, figsize=(14, 17))
    min_ = 100
    max_ = 0
    for classifier in classifiers:
        try:
            l = dic[classifier.__name__]
        except:
            continue
        l = [x * 100 for x in l]
        max_ = max(l) if max(l) > max_ else max_
        min_ = min(l) if min(l) < min_ else min_
    for index, ax in enumerate(axes.flatten()):
        try:
            l = dic[classifiers[index].__name__]
        except:
            continue
        l = [x * 100 for x in l]
        ax.plot(range(0, len(l)), l,
                marker='.', linestyle=linestyle, color=plots[index][0].get_color())
        ax.set_ylabel(y_label)
        ax.set_xticks([])
        ax.set_title(classifiers[index].__name__)
        ax.set_ylim([min_ - 0.5, max_ + 0.5])
        ax.grid(True)
    fig.suptitle(title, fontsize=16)
    # fig.tight_layout()
    if save:
        plt.savefig(f'{save_loc}/line_{name}.png', bbox_inches='tight')
        plt.savefig(f'{save_loc}/line_{name}.pdf', bbox_inches='tight')
    plt.show()


def plot_all(dics, names, y_labels, titles, linetype='solid'):
    loc = f'plots_{no_of_experiments}_{time.time()}'
    if save:
        mkdir(loc)

    X_train, X_test, y_train, y_test = splitting(features, target, state=range_start, verbose=verbose)
    if not plot_only:
        plot_roc(X_train, X_test, y_train, y_test, save_loc=loc)
    plot_confusion_matrix(X_train, X_test, y_train, y_test, save_loc=loc)

    for index, dic in enumerate(dics):
        plot_line(dic, names[index], y_label=y_labels[index], title=titles[index], linestyle=linetype, save_loc=loc)
        plot_bar(dic, names[index], y_label=y_labels[index], title=titles[index], save_loc=loc)

    if plot_only:
        plot_regions(X_train, X_test, y_train, y_test, save_loc=loc)


############################
# TESTING
############################
def evaluate(no_of_experiments):
    print(color.bold)
    with alive_bar(no_of_experiments, title='Evaluating', dual_line=True) as bar:
        for i in range(range_start, (range_start + no_of_experiments)):
            bar.text = f'-> Evaluating {i}...'
            # print(
            #       f'Evaluating {i} ({round(((i + 1 - range_start) * 100) /
            #       ((range_start + no_of_experiments) - range_start))}%)...')

            X_train, X_test, y_train, y_test = splitting(features, target, state=i, verbose=verbose)

            for classifier in classifiers:
                if verbose: print(f'Evaluating {classifier.__name__}...')
                score, probs, y_pred = classifier(X_train, X_test, y_train, y_test, verbose=verbose)
                try:
                    scores[classifier.__name__].append(score)
                except:
                    scores[classifier.__name__] = [score]
                try:
                    auc[classifier.__name__].append(roc_auc_score(y_test, probs[:, 1]))
                except:
                    auc[classifier.__name__] = [roc_auc_score(y_test, probs[:, 1])]
                try:
                    precision[classifier.__name__].append(precision_score(y_test, y_pred, pos_label='M'))
                except:
                    precision[classifier.__name__] = [precision_score(y_test, y_pred, pos_label='M')]
                try:
                    bal_accuracy_c[classifier.__name__].append(balanced_accuracy_score(y_test, y_pred, adjusted=True))
                except:
                    bal_accuracy_c[classifier.__name__] = [balanced_accuracy_score(y_test, y_pred, adjusted=True)]
                try:
                    bal_accuracy[classifier.__name__].append(balanced_accuracy_score(y_test, y_pred, adjusted=False))
                except:
                    bal_accuracy[classifier.__name__] = [balanced_accuracy_score(y_test, y_pred, adjusted=False)]
                try:
                    recall[classifier.__name__].append(recall_score(y_test, y_pred, pos_label='M'))
                except:
                    recall[classifier.__name__] = [recall_score(y_test, y_pred, pos_label='M')]
                try:
                    avg_precision[classifier.__name__].append(
                        average_precision_score(y_test, probs[:, 1], pos_label='M'))
                except:
                    avg_precision[classifier.__name__] = [average_precision_score(y_test, probs[:, 1], pos_label='M')]
                if verbose:
                    print(f'Score: {score} | probs: {probs} | y_pred: {y_pred}')
                    print(classification_report(y_test, y_pred))
            bar()
    print(color.end)


def create_results_table(sort_by_index=0):
    mean_scores = []
    mean_auc = []
    mean_precision = []
    mean_bal_accuracy = []
    mean_bal_accuracy_c = []
    mean_recall = []
    mean_avg_precision = []
    for func, l in scores.items():
        mean_scores.append(np.mean(l))
        mean_auc.append(np.mean(auc[func]))
        mean_precision.append(np.mean(precision[func]))
        mean_bal_accuracy.append(np.mean(bal_accuracy[func]))
        mean_bal_accuracy_c.append(np.mean(bal_accuracy_c[func]))
        mean_recall.append(np.mean(recall[func]))
        mean_avg_precision.append(np.mean(avg_precision[func]))

    mean_concat = {
        'Mean accuracy': mean_scores,
        'Mean balanced accuracy': mean_bal_accuracy,
        ' (chance-corrected)': mean_bal_accuracy_c,
        'Mean AUC': mean_auc,
        'Mean precision': mean_precision,
        'Mean recall': mean_recall,
        'Mean avg precision': mean_avg_precision
    }
    means = pd.DataFrame(mean_concat, index=scores.keys(),
                         columns=['Mean accuracy', 'Mean balanced accuracy', ' (chance-corrected)', 'Mean AUC',
                                  'Mean precision', 'Mean recall', 'Mean avg precision'])
    means.sort_values(by=means.columns[sort_by_index], ascending=False, inplace=True)
    print(f'{color.yellow}Sorted by {means.columns[sort_by_index]}{color.end}')

    return means


def create_test_tables():
    X_train, X_test, y_train, y_test = splitting(features, target, state=range_start, verbose=verbose)
    for classifier in classifiers:
        _, _, _ = classifier(X_train, X_test, y_train, y_test, verbose=verbose, pickle=True)

    ttest = pd.DataFrame([], columns=scores.keys(), index=scores.keys())
    pvaluest = pd.DataFrame([], columns=scores.keys(), index=scores.keys())
    ftest = pd.DataFrame([], columns=scores.keys(), index=scores.keys())
    pvaluesf = pd.DataFrame([], columns=scores.keys(), index=scores.keys())
    tttest = pd.DataFrame([], columns=scores.keys(), index=scores.keys())
    pvaluestt = pd.DataFrame([], columns=scores.keys(), index=scores.keys())
    with alive_bar((ttest.shape[0] * ttest.shape[1]), title='Performing T-tests and F-tests', dual_line=True) as bar:
        for rowindex, row in ttest.iterrows():
            for colindex, value in row.items():
                bar.text = f'-> Pitting {rowindex} against {colindex}...'
                ttest_res = t_test(rowindex, colindex)
                ftest_res = comb_f_test(True, rowindex, colindex, MinMaxScaler().fit_transform(features), target)
                tttest_res = comb_f_test(False, rowindex, colindex, MinMaxScaler().fit_transform(features), target)
                ttest.loc[rowindex, colindex] = ttest_res[1]
                pvaluest.loc[rowindex, colindex] = ttest_res[0]
                ftest.loc[rowindex, colindex] = ftest_res[1]
                pvaluesf.loc[rowindex, colindex] = ftest_res[0]
                tttest.loc[rowindex, colindex] = tttest_res[1]
                pvaluestt.loc[rowindex, colindex] = tttest_res[0]
                bar()

    return ttest, pvaluest, ftest, pvaluesf, tttest, pvaluestt


# These dictionaries store all metrics
# Structure: { 'MLP': [ 0.5748, 0.93828] }
# Note: classifiers are stored as strings here, not function-objects
scores = {}  # Accuracy scores
auc = {}
precision = {}
recall = {}
avg_precision = {}
bal_accuracy = {}  # Balanced accuracy respects the class imbalance
bal_accuracy_c = {}  # Balanced accuracy + adjusts for chance

############################
# SETTINGS
############################
verbose = False  # If True, print extra information for debugging and such
plot = True  # Whether to plot the ROC, confusion matrices, bar plot(s) and line diagram(s)
plot_only = True  # If true, do not run experiments, only create line/bar plots with previous data and current settings
save = False  # If true, save all plots in a /plots_{no_of_experiments}_{unix_time} folder. Plots will still be shown
compare_dummy = True  # If true, additional dummy classifiers are shown in the results and plots
perform_stat_tests = False
apply_pca = True
pca_components = 6
no_of_experiments = 10000  # 1.4 evaluations per second (on Thijs' machine, all cores)
n_jobs = -1  # -1 uses all cpu cores, 1 is default
pvalue_threshold = 0.05  # If calculated p-value is below threshold, it is considered statistically valid (reject n0)
csv_path = 'data.csv'
csv_separator = ','  # Default for .csv files is ','

range_start = random.randint(0, 9999)  # is seed for ROC/CM plot's train_test_split, as well as combined F-tests
# range_start = 0  # Enable for reproducible results

metrics_to_plot = [scores, bal_accuracy_c]
linetype = 'None'  # 'None' or 'solid'  # 'None' for many experiments, 'solid' for a few

metric_names = ['accuracy',  # Necessary for file naming if save = True
                'balanced_accuracy_c']
metric_y_labels = ['Accuracy score (%)',
                   'Balanced Accuracy score']
metric_titles = ['Accuracy of each classifier, each evaluation',  # Only visible in joined plots
                 'Chance-corrected Balanced Accuracy score for each classifier, each evaluation']

classifiers = [Logistic_Regression, MLP, KNN, Random_Forest, Gaussian_Naive_Bayes, Support_Vector_Machine,
               Dummy_Stratified, Dummy_Uniform]

classifiers_plot = [MLP, Gaussian_Naive_Bayes, Dummy_Stratified]


############################
# MAIN CODE
############################
def main():
    global no_of_experiments, features, target, scores, auc, precision, recall, avg_precision, bal_accuracy, \
        bal_accuracy_c
    features, target = import_csv(path=csv_path, separator=csv_separator, verbose=verbose)

    if not compare_dummy:
        classifiers.remove(Dummy_Stratified)
        classifiers.remove(Dummy_Uniform)

    while no_of_experiments < 2:  # Otherwise errors occur
        no_of_experiments += 1

    if not plot_only:
        evaluate(no_of_experiments)

        with open('metrics.pkl', 'wb') as file:
            dump([scores, auc, precision, recall, avg_precision, bal_accuracy, bal_accuracy_c], file)

        print('\n###############################\n'
              'TESTING FINISHED, RESULTS BELOW'
              '\n###############################\n')
        print(f'{color.yellow}{no_of_experiments} evaluations (starting from {range_start}){color.end}\n')

        # ['Mean accuracy', 'Mean balanced accuracy', ' (chance-corrected)', 'Mean AUC',
        #   'Mean precision', 'Mean recall', 'Mean avg precision']
        print(create_results_table(sort_by_index=0).to_string(), '\n')

        if perform_stat_tests:
            ttest, pvaluest, ftest, pvaluesf, tttest, pvaluestt = create_test_tables()

            print(
                f'\n{color.yellow}Which algorithm is statistically better compared to another? Simple T-test table '
                f'below:{color.end}')
            print(ttest.to_string(), '\n')
            print(f'{color.yellow}The corresponding p-values are as follows,'
                  f' if value < {pvalue_threshold} it is considered valid{color.end}')
            print(pvaluest.to_string(), '\n')

            print(
                f'{color.yellow}Which algorithm is statistically better compared to another? F-test table below:{color.end}')
            print(ftest.to_string(), '\n')
            print(f'{color.yellow}The corresponding p-values are as follows,'
                  f' if value < {pvalue_threshold} it is considered valid{color.end}')
            print(pvaluesf.to_string(), '\n')

            print(
                f'{color.yellow}Which algorithm is statistically better compared to another? 5x2 T-test table below:{color.end}')
            print(tttest.to_string(), '\n')
            print(f'{color.yellow}The corresponding p-values are as follows,'
                  f' if value < {pvalue_threshold} it is considered valid{color.end}')
            print(pvaluestt.to_string(), '\n')

    if plot:
        if plot_only:
            print(f'{color.yellow}{color.bold}\nThe script is running in plot-only mode,'
                  f' these plots are with the same data of the previous run!{color.end}')
            with open('metrics.pkl', 'rb') as file:
                dic_list = load(file)
                scores = dic_list[0]
                auc = dic_list[1]
                precision = dic_list[2]
                recall = dic_list[3]
                avg_precision = dic_list[4]
                bal_accuracy = dic_list[5]
                bal_accuracy_c = dic_list[6]
            metrics_to_plot_2 = []
            for index, _ in enumerate(metrics_to_plot):
                metrics_to_plot_2.append(dic_list[index])

            plot_all(metrics_to_plot_2, metric_names, metric_y_labels, metric_titles, linetype=linetype)
        else:
            plot_all(metrics_to_plot, metric_names, metric_y_labels, metric_titles, linetype=linetype)

    try:
        if no_of_experiments > 199 and not plot_only:
            notify_thais(no_of_experiments)
    except:
        pass


try:
    # for j in range(0, 5):
    main()
except Exception as main_error:
    print(f'{color.red}\n{traceback.format_exc()}{color.end}')
    try:
        notify_thais(f'{traceback.format_exc()}')
    except:
        pass
