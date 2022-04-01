import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import sklearn.metrics

dataset = pd.read_csv('data.csv')
dataset.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

y = dataset.diagnosis
X = dataset.drop(['diagnosis'], axis=1)

# drop correlated features (but it doesn't matter for the score)
#X.drop(['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se',
#                    'radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se',
 #                   'concave points_se','texture_worst','area_worst'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# scaling data with Yeo Johnson
from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer(method = 'yeo-johnson')

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# get gaussian NB model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)

ax = sn.heatmap(cm, annot=True, cmap='Blues', fmt='d')
ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

plt.savefig('cm.png')
#plt.show()

print(ac)
print(cm)

# ROC curve
from sklearn.metrics import roc_curve, auc
y_model = classifier.predict_proba(X_test)
curve = sklearn.metrics.roc_curve(y_test, y_model[:,1], pos_label = 'M')
# AUC
auc_model = auc(curve[0], curve[1])
print(f'auc: {auc_model}')
plt.plot(curve[0], curve[1], label='Gaussian Naive Bayes (area = %0.2f)'% auc_model)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')

plt.legend()
plt.savefig('ROC.png')

# Cross validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer

scores = cross_val_score(classifier, X, y, cv=10, scoring='roc_auc')

print('scores per fold ', scores)
print('  mean score    ', np.mean(scores))
print('  standard dev. ', np.std(scores))
