import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import math
from sklearn.model_selection import GridSearchCV
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV


def knn(x_train, x_test, y_train, y_test):

    # testing different distance metrics
    dist_metrics = ['euclidean', 'manhattan', 'l2', 'chebyshev', 'minkowski', 'cosine']

    # using grid search to find optimal k value
    for metric in dist_metrics:
        knn2 = KNeighborsClassifier(metric= metric)
        param_grid = {'n_neighbors': np.arange(1,40)}
        knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
        knn_gscv.fit(x_train, y_train)
        # print(knn_gscv.best_params_)
        # print(knn_gscv.best_score_)

    # for graphs of optimal k value
    all_scores = []
    for metric in dist_metrics:
        mean = []
        for k in range(1, 41):
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(x_train, y_train)
            score = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')
            print(f'Evaluating {metric} with k={k}\nScore: {score.mean()}')
            mean.append(score.mean())
        all_scores.append(mean)

    max_scores = []

    # finding the max_score across all different distance metrics
    for score_list in all_scores:
        max_score = np.max(score_list)
        max_scores.append(max_score)
        max_s = np.max(max_scores)
    # print(f'max {max_s}')

    ks = list(range(1, 41))  # x-axis of the k_val_plots

    k_val_plots = True  # change to true if you want to see plot
    if k_val_plots:
        fig, axs = plt.subplots(6)
        fig.suptitle('Plots of distance metrics and k values')
        for i in range(len(dist_metrics)):
            axs[i].plot(ks, all_scores[i])
            axs[i].set_title(dist_metrics[i])
        plt.show()

    # instantiating the classifier
    k = 4  # optimal k according to grid search
    classifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean')  # euclidean provided best accuracy
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    # ( cross validated ) accuracy & confusion matrix

    print(classification_report(y_test, y_pred))
    scores = cross_val_score(classifier, x_train, y_train, cv=5, scoring='accuracy')
    print(f'Cross validated accuracy on training set: {scores.mean()}')

    # confusion matrix
    cf = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf, annot=True, fmt='d')
    conf_plot = True  # change to true if you want to see plot
    if conf_plot:
        plt.show()

    # ROC/AUC ***
    prob = classifier.predict_proba(x_test)
    curve_knn = sklearn.metrics.roc_curve(y_test, prob[:, 1], pos_label='M')
    auc_knn = auc(curve_knn[0], curve_knn[1])
    plt.plot(curve_knn[0], curve_knn[1], label='knn (area = %0.2f)' % auc_knn)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    roc_plot = True  # change to true if you want to see plot
    if roc_plot:
        plt.show()


def split_and_scale(file):
    dataset = pd.read_csv(file)

    X = dataset.iloc[:, 2:32].values
    y = dataset.iloc[:, 1].values

    state = 999  # KNN is highly sensitive to random state

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=state)

    # knn is distance based so needs scaling
    scaler = MinMaxScaler() # MinMaxScalar provided optimal scaling
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


# Main
scaled_data = split_and_scale('data.csv')
knn(scaled_data[0], scaled_data[1], scaled_data[2], scaled_data[3])
