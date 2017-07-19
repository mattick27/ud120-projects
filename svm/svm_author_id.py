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
from sklearn.svm import LinearSVC
"""
features_train, features_test, labels_train, labels_test = preprocess()

clf = LinearSVC()

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

accuracy = clf.score(features_test, labels_test)
"""
############# Kernel linear 0.984072810011
from sklearn import svm

#### change parameter to number cause found error can't slice by parameter on spyder3
features_train = features_train[:158] 
labels_train = labels_train[:158] 

print('start train')
"""
linear_svc = svm.SVC(kernel = 'linear')
linear_svc.fit(features_train,labels_train)
pred = linear_svc.predict(features_test)
accuracy = linear_svc.score(features_test,labels_test)
print(accuracy)
print(linear_svc.kernel)
"""

############## rbf 0.492036405006

rbf_svc = svm.SVC(kernel = 'rbf')
rbf_svc.fit(features_train,labels_train)
pred = rbf_svc.predict(features_test)
accuracy = rbf_svc.score(features_test,labels_test)
print(accuracy)
print(rbf_svc.kernel)

#########################################################


