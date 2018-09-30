#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=10000.0)

# To reduce the size of training set and optimize training time, uncomment the below
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
t0 = time()
clf.fit(features_train, labels_train)
print ('Time for training = ', round(time()-t0, 3), 's')

t0 = time()
pred_test = clf.predict(features_test)
print ('Time for testing = ', round(time()-t0, 3), 's')

print(pred_test[10])
print(pred_test[26])
print(pred_test[50])

import numpy as np
chris_mails = np.where(pred_test==1)[0]
print('Mails from Chris = ', len(chris_mails))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred_test)

print ('Accuracy = ', accuracy)

#########################################################


