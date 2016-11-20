from __future__ import print_function
from thesis_imports import *
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X, y = iris.data, iris.target
# X is of shape (150, 4)
print (X)
# y is of shape (150,1)
print (y)
array = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
print (array)
