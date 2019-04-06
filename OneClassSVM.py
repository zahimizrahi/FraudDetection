from sklearn.datasets import fetch_mldata
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter


class OneClassSVM:

    def __init__(self):
        return

    def oneclass_svm_baseline(self, data, targets):
        X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=0)
        # build a one class svm model
        model = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
        model.fit(X_train)
        # make prediction
        for i in range(10):
            predictions = [int(a) for a in model.predict(X_test)]
            num_corr = sum(int(a == 1) for a in predictions)
            print "   %d   " % i,
            if i == 0:
                print "%d of %d" % (num_corr, len(predictions))
            else:
                print "%d of %d" % (len(predictions) - num_corr, len(predictions))
            pass

