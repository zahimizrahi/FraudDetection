import numpy

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.svm import OneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn import model_selection
import pickle

SEED = 10

class ClassificationModel:
    feature_select_output_dir = 'outputs/FeatureSelector/'

    def __init__(self, user_num, n=2, type='ngram', n_features=250):
        # TODO - use all users data
        df = pd.read_csv(self.feature_select_output_dir + '{}-{}-user{}.csv'.format(type, n, user_num))
        self.arr = df.values
        X = self.arr[:, 0:n_features]
        Y = df.pop('Label').values
        if user_num < 10:
            # pop the target column
            #print (df.gr oupby('Label').size())
            #df = df[df['Label'].isin(['0','1'])]
            self.x_train, self.x_test, self.y_train, self.y_test = X[0:100], X[100:], Y[0:100], Y[100:]
            print self.x_train
                #model_selection.train_test_split(X,Y,test_size=0.3,random_state=7)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = X[0:25], X[25:50], Y[0:25], Y[25:50]

    # def logisticRegression(self):
    #     logisticRegr = LogisticRegression()
    #     logisticRegr.fit(self.x_train, self.y_train)
    #     predictions = logisticRegr.predict(self.x_test)
    #     score = logisticRegr.score(self.x_test, self.y_test)
    #     print "logisticRegression score = "
    #     print score
    #
    # def oneclassSvm(self):
    #     # build a one class svm model
    #     model = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
    #     model.fit(self.x_train, self.y_train)
    #     score = model.score_samples(self.x_test)
    #     print "oneclassSvm score = "
    #     print score
    #
    # def decisionTree(self):
    #     model = tree.DecisionTreeClassifier()
    #     model.fit(self.x_train, self.y_train)
    #     score = model.score(self.x_test, self.y_test)
    #     print "decisionTree score = "
    #     print score

    def compare_models(self):
        # prepare models
        models = [('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()),
               #('LogisticRegression', LogisticRegression(solver='liblinear')),
                  ('MLPClassifier', MLPClassifier()),
                  ('KNeighbors', KNeighborsClassifier()),
                  ('GaussianNaiveBayes', GaussianNB()),
                  ('DecisionTree', DecisionTreeClassifier()),
                  ('RandomForest', RandomForestClassifier(n_estimators=100)),
                  ('AdaBoost', AdaBoostClassifier()),
                  #('GradientBoosting', GradientBoostingClassifier()),
                  #('SVC', SVC(gamma='scale')),
                  #('LinearSVC', LinearSVC()),
                  ('OneClassSVM', OneClassSVM(gamma='scale', nu=0.1))]
        # evaluate each model in turn
        results = []
        names = []
        #special_list = ['LogisticRegression', 'OneClassSVM']
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=0)
            cv_results = model_selection.cross_val_score(model, self.x_test, self.y_test, cv=kfold, scoring='accuracy')
            results.append((name, cv_results.mean(), cv_results.std()))
            names.append(name)
            msg = '{}-{}-{}'.format(name, cv_results.mean(), cv_results.std())
            print msg

                #if name in special_list:
            #    model.fit(self.x_train)
            #else:
            #    model.fit(self.x_train, self.y_train)
            #preds = model.predict(self.x_test)
            #msg = "Name: {}    Accuracy: {:.4%}".format(name, accuracy_score(self.y_test, preds))
            #print "X_train: {}".format(len(self.x_train))
            #print "Y_train: {}".format(len(self.y_train))
            #model.fit(self.x_train, self.y_test)
            #preds = model.predict(self.x_test)
            #print preds
        # boxplot algorithm comparison
        #fig = plt.figure()
        #fig.suptitle('Algorithm Comparison')
        #ax = fig.add_subplot(111)
        #plt.boxplot(results)
        #ax.set_xticklabels(names)
        #plt.show()
        return results

    def predictLabels(self, user_num, n=2, type='ngram', n_features=250, x_train=None, y_train=None):
        # Finalize model
        model = RandomForestClassifier(n_estimators=100)
        model.fit(x_train, y_train)

        # Save model
        filename = 'Final_Model.sav'
        pickle.dump(model, open(filename, 'wb'))

        # Load model and use it to make new predictions
        loaded_model = pickle.load(open(filename, 'rb'))

        # Load test dataset
        df = pd.read_csv(self.feature_select_output_dir + '{}-{}-user{}.csv'.format(type, n, user_num))
        self.arr = df.values
        #FixMe - fix error
        #X = self.arr[:, 0:n_features]
        #pred = model.predict(X[50:])
        #print(pred)



