from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection


class ClassificationAlgo:
    feature_select_output_dir = 'outputs/FeatureSelector/'

    def __init__(self, user_num, n=2, type='ngram'):
        # TODO - use all users data
        df = pd.read_csv(self.feature_select_output_dir + '{}-{}-user{}.csv'.format(type, n, user_num))
        # pop the target column
        Y = df.pop('Label').values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(df, Y, test_size=0.25, random_state=0)

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
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        # evaluate each model in turn
        results = []
        names = []
        scoring = 'accuracy'
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=7)
            cv_results = model_selection.cross_val_score(model, self.x_test, self.y_test, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

