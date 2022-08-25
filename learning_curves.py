import matplotlib.pyplot as plt
import numpy as np

def accuracy(predictions, ytest):
    # correct decisions / total decisions
    s = 0
    for i in range(len(predictions)):
        if predictions[i] == ytest[i]:
            s += 1

    return s / len(predictions)


def precision(predictions, ytest):
    # find true positives and false positives
    tp = 0
    fp = 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and ytest[i] == 1:
            tp += 1
        if predictions[i] == 1 and ytest[i] == 0:
            fp += 1
    return tp / (tp + fp)

def recall(predictions, ytest):
    # find true positives and false negatives
    tp = 0
    fn = 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and ytest[i] == 1:
            tp += 1
        if predictions[i] == 0 and ytest[i] == 1:
            fn += 1
    return tp / (tp + fn)

def F_measure(predictions, ytest, b=1):
    p = precision(predictions,ytest)
    r = recall(predictions,ytest)   
    return  (b**2 + 1) * ((p*r) / ((b**2)*p + r))
