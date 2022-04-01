from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, svm
from sklearn.metrics import plot_roc_curve, confusion_matrix, plot_confusion_matrix


def support_vector_machine(file, labels):

    x_train, x_test, y_train, y_test = train_test_split(file, labels, test_size=0.3, random_state=42)

    # SVM needs normalization
    x_train = normalize(x_train, axis=0)
    x_test = normalize(x_test, axis=0)

    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    print("Accuracy:", metrics.accuracy_score(y_test, predictions))

    # confusion matrix
    conf = confusion_matrix(y_test, predictions)

    plot = False
    if plot:
        sns.heatmap(conf, fmt='d', annot=True)
        plt.show()

    # plot ROC plot
    plot = False
    if plot:
        plot_roc_curve(clf ,x_test, y_test)
        plt.show()



def random_forest(file, labels):

    x_train, x_test, y_train, y_test = train_test_split(file, labels, test_size=0.3, random_state=42)

    print('Training Features Shape:', x_train.shape)
    print('Training Labels Shape:', y_train.shape)
    print('Testing Features Shape:', x_test.shape)
    print('Testing Labels Shape:', y_test.shape)

    # lab_enc = preprocessing.LabelEncoder()
    # encoded = lab_enc.fit_transform(x_train)

    rf = RandomForestClassifier(n_estimators=1000, random_state=43)
    rf.fit(x_train, y_train)

    predictions = rf.predict(x_test)

    # plot confusion matrix
    conf = confusion_matrix(y_test, predictions)

    plot = False
    if plot:
        sns.heatmap(conf, fmt='d', annot=True)
        plt.show()

    # plot ROC plot
    plot = False
    if plot:
        plot_roc_curve(rf ,x_test, y_test)
        plt.show()


# feature selection (this is for random forrest and SVM without doing PCA)

def feature_select(file):

    # drop highly correlated features
    file = file.drop(['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se',
                    'radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se',
                    'concave points_se','texture_worst','area_worst'], axis=1)

    # drop labels and store them in a value
    labels = file.diagnosis
    file = file.drop(['diagnosis'], axis=1)
    print(file.head())
    print(labels)

    # plot a new correlation matrix after feature selection
    plot = False
    if plot:
        corr = file.corr()
        sns.heatmap(corr, fmt = '0.1f', annot=True)
        plt.show()

    # Random forrest
    random_forest(file, labels)

    # SVM
    support_vector_machine(file, labels)


# Data exploration
def visualize(file):
    features = file.columns
    labels = file.diagnosis
    # print(features)

    # plot labels
    plot = False
    if plot:
        ax = sns.countplot(file['diagnosis'], label='Count')
        B, M = file['diagnosis'].value_counts()
        plt.show()

    # print(file.describe())

    # plot distribution of first 10 variables
    plot = False
    if plot:
        sns.set(style="whitegrid", palette="muted")
        data_n_2 = (file - file.mean()) / (file.std())
        data = pd.concat([labels, data_n_2.iloc[:, 0:10]], axis=1)
        print(data)
        data = pd.melt(data, id_vars='diagnosis', var_name='features', value_name='value')

        plt.figure(figsize=(10, 10))
        tic = time.time()
        sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
        plt.xticks(rotation=90)
        plt.show()

    # plot correlation matrix
    plot = False
    if plot:
        plt.figure(figsize=(13, 10))
        corr = file.corr()
        # print(corr)
        high_corr = {}
        # store features with high correlation in a dictionary to be excluded later on
        for key, value in corr.items():
            for key1, value1 in value.items():
                if value1 > 0.85 and key != key1:
                    # print(key,key1,value1)
                    if key not in high_corr.keys():
                        high_corr[key] = []
                        high_corr[key].append(key1)
                    else:
                        high_corr[key].append(key1)

        # print(high_corr)

        sns.heatmap(corr, fmt='.1f')
        plt.show()

    # plt.hist(file['diagnosis'], bins=2,color=file['diagnosis'].value_counts())
    # file['diagnosis'].value_counts().plot(kind = 'bar')
    # plt.show()


# pca + random forest and SVM
def pca(file):

    file2 = file
    y = file2.diagnosis     # get labels
    file2 = file2.drop(['diagnosis'], axis=1)     # drop labels from dataframe

    # scaling before PCA
    scaling = True
    if scaling:
        file2 = StandardScaler().fit_transform(file2)
        file2 = pd.DataFrame(file2)
        print(np.mean(file2), np.std(file2))

    # log transforming + normalizing before PCA
    log_transform = False
    if log_transform:
        file2 = np.log10(file2 + 0.001)
        file2 = normalize(file2, axis=0)
        file2 = pd.DataFrame(file2)
        print(np.mean(file2), np.std(file2))
        # print(file2)

    plot = False
    if plot:
        file2.hist()
        plt.show()

    # create PCA model
    pca = PCA(n_components=6)
    pca.fit(file2, y)

    # get % of variance explained by each PC
    print(pca.explained_variance_ratio_.cumsum())

    # reduce dimensions to 6 PCs
    x_reduced = pca.transform(file2)
    x_train, x_test, y_train, y_test = train_test_split(x_reduced, y, test_size=0.3, random_state=42)

    # perform svm on 2 PCs
    model = svm.SVC(kernel='linear')
    model.fit(x_train, y_train)

    prediction = model.predict(x_test)

    # plot svm results
    plot = False
    if plot:
        plot_roc_curve(model, x_test, y_test)
        plt.show()
        conf = confusion_matrix(y_test, prediction)
        sns.heatmap(conf, annot=True, fmt='d')
        plt.show()

    #       **** Random forest on first 6 PCs ****
    # random_forest(x_reduced,y)


def main():
    # f_in = str(argv[1])
    file = pd.read_csv('data.csv')
    file.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    features = file.columns
    # file.drop(columns=features[list(range(10, 30))], axis=1, inplace=True)
    # file.diagnosis.replace(['B','M'], [0,1], inplace=True)

    # call functions. feature_select function calls the classifiers

    pca(file)
    visualize(file)
    feature_select(file)


if __name__ == '__main__':
    main()
