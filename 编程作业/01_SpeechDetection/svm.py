#!/usr/bin/env python3
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import time

def readData(filename):
    yApp = []
    xApp = []
    with open(filename, 'rb') as f:
        data = f.readlines()
    for l in data:
        t = l.split(b' ')
        t = [float(n) for n in t]
        yApp.append(int(t[-1]))
        xApp.append(t[:-1])
    print(len(yApp))
    print(len(xApp))
    return yApp, xApp

featureLabel, featureData = readData('training.data')
featureData = MinMaxScaler().fit_transform(featureData)
xTraining, xTesting, yTraining, yTesting = train_test_split(featureData, featureLabel, test_size=0.1)
print(len(xTraining))
print(len(yTesting))
print(len(xTraining))
print(len(yTesting))

SVM = SVC(C=2, kernel='rbf', degree=3, gamma=2)
SVM.fit(xTraining, yTraining)
score = SVM.score(xTraining, yTraining)
predict_SVM = SVM.predict(xTesting)
accuracy_SVM = metrics.accuracy_score(yTesting, predict_SVM)
print(score, accuracy_SVM)

