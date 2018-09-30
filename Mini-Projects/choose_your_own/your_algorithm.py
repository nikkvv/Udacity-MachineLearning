#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
#plt.scatter(grade_slow, bumpy_slow, color = "b", label="slow")
plt.scatter(bumpy_slow, grade_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from time import time

method = 'knn'
#method = 'random_forest'
#method = 'adaboost'

if method=='knn':
    # K-Nearest Neighbors Classifier
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=22, weights='uniform')
elif method=='random_forest':
    # Random forest
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=50, min_samples_split=5)
elif method=='adaboost':
    # Adaboost
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=20) , n_estimators=100)
else:
    pass

try:
    t0 = time()
    clf.fit(features_train, labels_train)
    print('Training time = ', round(time()-t0, 3), 's')
    t0 = time()
    pred_test = clf.predict(features_test)
    print('Testing time = ', round(time()-t0, 3), 's')
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels_test, pred_test)
    print('Accuracy = ', accuracy)

    prettyPicture(clf, features_test, labels_test)
except:
    pass


