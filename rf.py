import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, auc, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PowerTransformer
import requests

data = pd.read_csv('data.csv')
features = data.iloc[:, 2:32].values
target = data.iloc[:, 1].values
scaler = PowerTransformer()


def display(classifier):  # auc stolen from Lieve
    global y_predicted
    y_predicted = classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_predicted)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()
    prob = classifier.predict_proba(x_test)
    curve_knn = roc_curve(y_test, prob[:, 1], pos_label='M')
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


single_test = True
if single_test:
    no_of_features = ["sqrt"]
    no_of_trees = [1000]
else:
    testing = True  # for a quick run
    if testing:
        no_of_features = ["sqrt", "log2", 3]
        no_of_trees = [10, 20]
    else:
        no_of_features = ["sqrt", "log2", 3, 5, 7, 10, None]
        no_of_trees = [10, 20, 50, 100, 200, 500, 1000]


def results(x_train, x_test, y_train, y_test, no_of_trees, no_of_features):
    table = {}
    for i in no_of_trees:
        temp = []
        for j in no_of_features:
            train(x_train, x_test, y_train, y_test, i, j)
            temp.append(cv_average)
            display(forest)
        table[i] = temp
    global df
    df = pd.DataFrame(data=table)
    df.set_index([no_of_features], inplace=True)
    print(df)


def train(x_train, x_test, y_train, y_test, no_of_trees, no_of_features):
    scaler.fit(x_train)
    global forest
    forest = RandomForestClassifier(n_estimators=no_of_trees, max_features=no_of_features)  # default n of trees is 100
    forest.fit(x_train, y_train)
    folds = 5
    global cv_average
    cv_array = cross_val_score(forest, x_train, y_train, cv=folds, verbose=0)
    cv_average = cv_array.mean()


x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3,
                                                    random_state=10 * round(random.random()))
results(x_train, x_test, y_train, y_test, no_of_trees, no_of_features)

