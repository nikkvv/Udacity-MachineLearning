#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

train_feats, test_feats, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(train_feats, train_labels)
print("Accuracy = ", clf.score(test_feats, test_labels))

import numpy as np
pred_test = clf.predict(test_feats)
print('People in test set = ', len(test_feats))
print('Total Predicted POIs = ', sum(pred_test))

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
#print(classification_report(test_labels, pred_test, target_names=['notPOI', 'POI']))
print(confusion_matrix(test_labels, pred_test, labels=range(2)))
print('Precision')
print(precision_score(test_labels, pred_test))
print('Recall')
print(recall_score(test_labels, pred_test))