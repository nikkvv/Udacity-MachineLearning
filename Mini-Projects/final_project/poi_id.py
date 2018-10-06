#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = [
    'poi', 
    'total_payments', 
    'exercised_stock_options',  
    'expenses', 
    'from_this_person_to_poi'
]
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

#clf = GaussianNB()

#clf = SVC()

clf = DecisionTreeClassifier(min_samples_split=3, random_state=42, criterion='entropy')

#clf = RandomForestClassifier(n_estimators=20, 
#                            min_samples_split=2, 
#                            random_state=42, 
#                            criterion='entropy')

#clf = AdaBoostClassifier(random_state=42)
print clf

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print('Training size = ', len(features_train))
print('Testing size = ', len(features_test))

clf.fit(features_train, labels_train)
pred_test = clf.predict(features_test)
pred_train = clf.predict(features_train)

from sklearn.metrics import precision_score, recall_score, accuracy_score
print('Train Accuracy = ', accuracy_score(labels_train, pred_train))
print('Test Accuracy = ', accuracy_score(labels_test, pred_test))
print('Train Precision = ', precision_score(labels_train, pred_train))
print('Test Precision = ', precision_score(labels_test, pred_test))
print('Train Recall = ', recall_score(labels_train, pred_train))
print('Test Recall = ', recall_score(labels_test, pred_test))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

print('Feature Importance = ', clf.feature_importances_)