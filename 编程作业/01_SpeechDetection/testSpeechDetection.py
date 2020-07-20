#!/usr/bin/env python3
from libsvm.svmutil import *
from operator import itemgetter
import numpy as np
import random
import matplotlib.pyplot as plt

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
    return yApp, xApp

def dealWithData(yApp, xApp, trainingDataLength):
    xTraining = []
    yTraining = []
    xTesting = []
    yTesting = []
    idxs = list(range(len(xApp)))
    random.shuffle(idxs)
    for i in range(trainingDataLength):
        xTraining.append(xApp[idxs[i]])
        yTraining.append(yApp[idxs[i]])
    for i in range(trainingDataLength, len(xApp)):
        xTesting.append(xApp[idxs[i]])
        yTesting.append(yApp[idxs[i]])
    '''
    avgX = np.mean(np.mat(xTraining), axis=0).tolist()[0]
    stdX = np.std(np.mat(xTraining), axis=0).tolist()[0]
    print('[*] avgX = ' + str(avgX))
    print('[*] stdX = ' + str(stdX))
    for data in xTraining:
        for i in range(len(data)):
            data[i] = (data[i] - avgX[i]) / stdX[i]
    for data in xTesting:
        for i in range(len(data)):
            data[i] = (data[i] - avgX[i]) / stdX[i]
    '''
    return yTraining, xTraining, yTesting, xTesting

def trainingModel(label, data, modelFilename):
    CScale = [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
    gammaScale = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3]
    maxRecognitionRate = 0
    maxC = 0
    maxGamma = 0
    for C in CScale:
        C_ = pow(2, C)
        for gamma in gammaScale:
            gamma_ = pow(2, gamma)
            cmd = '-t 2 -c ' + str(C_) + ' -g ' + str(gamma_) + ' -v 5 -q'
            recognitionRate = svm_train(label, data, cmd)
            if recognitionRate > maxRecognitionRate:
                maxRecognitionRate = recognitionRate
                maxC = C
                maxGamma = gamma
    n = 10
    minCScale = 0.5 * (min(-5, maxC) + maxC)
    maxCScale = 0.5 * (max(15, maxC) + maxC)
    newCScale = np.arange(minCScale, maxCScale+1, (maxCScale-minCScale)/n)
    print('[*] newCScale = ' + str(newCScale))
    minGammaScale = 0.5 * (min(-15, maxGamma) + maxGamma)
    maxGammaScale = 0.5 * (max(3, maxGamma) + maxGamma)
    newGammaScale = np.arange(minGammaScale, maxGammaScale+1, (maxGammaScale-minGammaScale)/n)
    print('[*] newGammaScale = ' + str(newGammaScale))
    for C in newCScale:
        C_ = pow(2, C)
        for gamma in newGammaScale:
            gamma_ = pow(2, gamma)
            cmd = '-t 2 -c ' + str(C_) + ' -g ' + str(gamma_) + ' -v 5 -q'
            recognitionRate = svm_train(label, data, cmd)
            if recognitionRate > maxRecognitionRate:
                maxRecognitionRate = recognitionRate
                maxC = C
                maxGamma = gamma
    print('[*] maxC = ' + str(maxC))
    print('[*] maxGamma = ' + str(maxGamma))
    C = pow(2, maxC)
    gamma = pow(2, maxGamma)
    cmd = '-t 2 -c ' + str(C) + ' -g ' + str(gamma)
    model = svm_train(label, data, cmd)
    svm_save_model(modelFilename, model)
    return model

def drawROC(yTesting, decisionValues):
    values, labels = [list(x) for x in zip(*sorted(zip(decisionValues, yTesting), key=itemgetter(0)))]
    truePositive = [0 for i in range(len(values) + 1)]
    trueNegative = [0 for i in range(len(values) + 1)]
    falsePositive = [0 for i in range(len(values) + 1)]
    falseNegative = [0 for i in range(len(values) + 1)]
    for i in range(len(values)):
        if labels[i] == 1:
            truePositive[0] += 1
        else:
            falsePositive[0] += 1
    for i in range(len(values)):
        if labels[i] == 1:
            truePositive[i + 1] = truePositive[i] - 1
            falsePositive[i + 1] = falsePositive[i]
        else:
            falsePositive[i + 1] = falsePositive[i] - 1
            truePositive[i + 1] = truePositive[i]
    truePositive = (np.array(truePositive) / truePositive[0]).tolist()
    falsePositive = (np.array(falsePositive) / falsePositive[0]).tolist()
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.plot(falsePositive, truePositive, color='blue')
    plt.plot([1,0], [0,1], color='red')
    plt.legend(['ROC', 'EER'])
    plt.show()

if __name__ == '__main__':
    yApp, xApp = readData('training.data')
    yTraining, xTraining, yTesting, xTesting = dealWithData(yApp, xApp, 5000)
    if input('Train or not? (y/n) ') == 'y':
        model = trainingModel(yTraining, xTraining, 'training.model')
    else:
        model = svm_load_model('training.model')
    yPred, accuracy, decisionValues = svm_predict(yTesting, xTesting, model)
    drawROC(yTesting, decisionValues)

