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
    feature_select_output_file = 'outputs/FeatureSelector/all.csv'
    def __init__(self, user_num, n=2, type='ngram', n_features=250):
        # TODO - use all users data
        df = pd.read_csv(self.feature_select_output_file)
        self.arr = df[df["User"]==user_num]
        X = self.arr.drop(columns=['Label', 'Segment','User', 'Unnamed: 0'])
        Y = self.arr.pop('Label').values
        if user_num < 10:
            # pop the target column
            #print (df.gr oupby('Label').size())
            #df = df[df['Label'].isin(['0','1'])]
            self.x_train, self.x_test, self.y_train, self.y_test = X[0:100], X[100:], Y[0:100], Y[100:]
            #print self.x_train
                #model_selection.train_test_split(X,Y,test_size=0.3,random_state=7)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = X[0:25], X[25:50], Y[0:25], Y[25:50]

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
        df = pd.read_csv(self.feature_select_output_file)
        self.arr = df[df["User"]==user_num]
        X = self.arr.drop(columns=['Label', 'Segment','User', 'Unnamed: 0'])
        pred = model.predict(X[50:])
        print(pred)



