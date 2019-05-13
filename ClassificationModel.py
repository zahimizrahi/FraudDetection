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

# from keras.models import Model, load_model
# from keras.layers import Input, Dense
# from keras.models import load_model
# from keras.models import save_model

SEED = 10


class ClassificationModel:
    feature_select_output_dir = 'outputs/FeatureSelector/'
    feature_select_output_file = 'outputs/FeatureSelector/all.csv'
    sample_submission_file = 'resources/sample_submission.csv'
    submission_file = 'final_submission.csv'
    autoencoder_output_dir = 'outputs/autoencoders/'

    def __init__(self, user_num, df):
        # TODO - use all users data
        self.user_num = user_num
        self.df = df.copy()
        self.sample_df = pd.read_csv(self.sample_submission_file)
        # Finalize model
        self.model = RandomForestClassifier(n_estimators=10)
        # model = LinearDiscriminantAnalysis()
        # model = RandomForestClassifier(n_neighbors=1, novelty=True, contamination='legacy')
        # model = OneClassSVM(nu=0.1, kernel='rbf', gamma=1e-5)
        # self.model = GaussianNB()
        # self.model = AdaBoostClassifier()
        # self.model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        self.model = OneClassSVM(nu=0.1, kernel='rbf', gamma=1e-5)
        if user_num < 10:
            self.arr = self.df[(self.df['User'] == user_num)]
        else:
            self.arr = self.df[(self.df['User'] == user_num) & (self.df['Label'] == 0)]
        self.y_train = self.arr.pop('Label')
        self.x_train = self.arr

    def optimize_parameters(self):
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['auto', 5, 2, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                             'nu': [0.1]}]
        # tuned_parameters = { 'novelty': [True], 'n_neighbors': range(1,21, 2),
        #                     'contamination': ['legacy', 0.1, 0.2, 0.3, 0.5]}
        # Split the dataset in two equal parts
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            self.x_train, self.y_train, test_size=0.5, random_state=0)

        score = 'accuracy'
        print "# Tuning hyper-parameters for %s" % score
        clf = GridSearchCV(OneClassSVM(), tuned_parameters, cv=5, scoring=score)
        clf.fit(X_train1, y_train1)

        print clf.best_params_

    def train_models(self, model=None):
        df = self.df.copy()
        normal_user_df = self.df.copy()
        normal_user_df = df[df["User"] == self.user_num].copy()
        normal_user_df['Label'] = 0
        other_user_df = self.df.copy()
        for other in range(40):
            if self.user_num != other:
                other_user_df = df[df['User'] == other].copy()
                other_user_df['Label'] = 1
            normal_user_df = normal_user_df.append(other_user_df)
        self.y_train = normal_user_df.pop('Label')
        self.x_train = normal_user_df.drop(columns=['Segment', 'User', 'User_index', 'Segment_index'])
        if model is None:
            return self.model.fit(self.x_train, self.y_train)
        else:
            return model.fit(self.x_train, self.y_train)

    def predictLabels(self):

        self.train_models()

        # Save model
        filename = 'Final_Model.sav'
        pickle.dump(self.model, open(filename, 'wb'))

        # Load model and use it to make new predictions
        loaded_model = pickle.load(open(filename, 'rb'))

        # Load test dataset
        df = pd.read_csv(self.feature_select_output_file)
        self.arr = df[df["User"] == self.user_num]
        # X = self.arr.drop(columns=['Label', 'Segment', 'User', 'Unnamed: 0'])
        X = self.arr.drop(columns=['Label', 'Segment', 'User', 'User_index', 'Segment_index'])
        X = numpy.array(X)
        preds = self.model.predict(X[50:])
        correct_preds = []

        for pred in preds:
            if pred == -1:
                correct_preds.append(1)
            else:
                correct_preds.append(0)

        print np.asarray(correct_preds)
        return np.asarray(correct_preds)


'''
        X_scores = self.model.negative_outlier_factor_

        plt.title("Local Outlier Factor (LOF)")
        plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
        # plot circles with radius proportional to the outlier scores
        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
        plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
                    facecolors='none', label='Outlier scores')
        plt.axis('tight')
        plt.xlim((0, 60))
        plt.ylim((0, 60))
        # plt.xlabel("prediction errors: %d" % (n_errors))
        legend = plt.legend(loc='upper left')
        legend.legendHandles[0]._sizes = [10]
        legend.legendHandles[1]._sizes = [20]
        # plt.show()

        return np.asarray(correct_preds)
'''





