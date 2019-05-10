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
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report
import pickle

SEED = 10


class ClassificationModel:
    feature_select_output_dir = 'outputs/FeatureSelector/'
    feature_select_output_file = 'outputs/FeatureSelector/all.csv'
    sample_submission_file = 'resources/sample_submission.csv'
    submission_file = 'final_submission.csv'

    def __init__(self, user_num):
        # TODO - use all users data
        self.user_num = user_num
        df = pd.read_csv(self.feature_select_output_file)
        self.sample_df = pd.read_csv(self.sample_submission_file)
        self.arr = df[(df["User"]==user_num) & (df["Label"]!=2)]
        if user_num < 10:
            self.arr_all = df[df["Label"] == 0]
        else:
            self.arr_all = df[(df["User"]!=user_num) & (df["Label"]==0)]
        X_All = self.arr_all.drop(columns=['Label', 'Segment','User', 'Unnamed: 0'])
        X = self.arr.drop(columns=['Label', 'Segment','User', 'Unnamed: 0'])
        Y_All = self.arr_all.pop('Label').values
        Y = self.arr.pop('Label').values
        if user_num < 10:
            self.x_train, self.x_test, self.y_train, self.y_test = X_All, X[50:], Y_All, Y[50:]
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = X_All, X, Y_All, Y

    def optimize_parameters(self):
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['auto', 5, 2, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                             'nu': [0.1]}]
        #tuned_parameters = { 'novelty': [True], 'n_neighbors': range(1,21, 2),
        #                     'contamination': ['legacy', 0.1, 0.2, 0.3, 0.5]}
        # Split the dataset in two equal parts
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            self.x_train, self.y_train, test_size=0.5, random_state=0)

        score = 'accuracy'
        print "# Tuning hyper-parameters for %s" % score
        clf = GridSearchCV(OneClassSVM(), tuned_parameters, cv=5, scoring=score)
        clf.fit(X_train1, y_train1)

        print clf.best_params_

    def compare_models(self):
        # prepare models
        models = [('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()),
                  ('MLPClassifier', MLPClassifier()),
                  ('KNeighbors', KNeighborsClassifier()),
                  ('GaussianNaiveBayes', GaussianNB()),
                  ('DecisionTree', DecisionTreeClassifier()),
                  ('RandomForest', RandomForestClassifier(n_estimators=100)),
                  ('AdaBoost', AdaBoostClassifier()),
                  ('OneClassSVM', OneClassSVM(gamma='scale', nu=0.1))]
        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
            kfold = model_selection.KFold(n_splits=5, random_state=0)
            cv_results = model_selection.cross_val_score(model, self.x_test, self.y_test, cv=kfold, scoring='accuracy')
            results.append((name, cv_results.mean(), cv_results.std()))
            names.append(name)
            msg = '{}-{}-{}'.format(name, cv_results.mean(), cv_results.std())
            print msg
        return results

    def predictLabels(self):
        # Finalize model
        #model = RandomForestClassifier(n_estimators=100)
        #model = LinearDiscriminantAnalysis()
        #model = LocalOutlierFactor(n_neighbors=1, novelty=True, contamination='legacy')
        model = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
        model.fit(self.x_train)

        # Save model
        filename = 'Final_Model.sav'
        pickle.dump(model, open(filename, 'wb'))

        # Load model and use it to make new predictions
        loaded_model = pickle.load(open(filename, 'rb'))

        # Load test dataset
        df = pd.read_csv(self.feature_select_output_file)
        self.arr = df[df["User"] == self.user_num]
        X = self.arr.drop(columns=['Label', 'Segment','User', 'Unnamed: 0'])
        preds = model.predict(X[50:])
        correct_preds = []

        for pred in preds:
            if pred == -1:
                correct_preds.append(1)
            else:
                correct_preds.append(0)

        print np.asarray(correct_preds)
        return np.asarray(correct_preds)





