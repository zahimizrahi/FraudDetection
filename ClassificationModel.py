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
from sklearn.ensemble import IsolationForest
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

pd.options.display.max_rows = 150
pd.options.display.max_columns = 150
# from keras.models import Model, load_model
# from keras.layers import Input, Dense
# from keras.models import load_model
# from keras.models import save_model

SEED = 10

pd.options.display.max_rows = 150
pd.options.display.max_columns = 150


class ClassificationModel:
    feature_select_output_dir = 'outputs/FeatureSelector/'
    feature_select_output_file = 'outputs/FeatureSelector/selected_all.csv'
    sample_submission_file = 'resources/sample_submission.csv'
    submission_file = 'final_submission.csv'
    autoencoder_output_dir = 'outputs/autoencoders/'

    def __init__(self, user_num, df, model=None):
        # TODO - use all users data
        self.user_num = user_num
        self.df = df.copy()
        self.sample_df = pd.read_csv(self.sample_submission_file)
        # Finalize model
        # self.model = IsolationForest(n_estimators=1, n_jobs=1, random_state=SEED, contamination=0.1)
        # self.model = LinearDiscriminantAnalysis()
        # self.model = IsolationForest(n_estimators=30,  contamination='legacy')
        # model = OneClassSVM(nu=0.1, kernel='rbf', gamma=1e-5)
        # self.model = GaussianNB()
        # self.model = SVC()
        # self.model = AdaBoostClassifier()
        # self.model = GradientBoostingClassifier(random_state=SEED)
        if model:
            self.model = model
        else:
            # self.model = LocalOutlierFactor(novelty=True)
            # self.model = LinearDiscriminantAnalysis()
            # self.model = IsolationForest(random_state=42, max_samples=100, contamination=0.1)
            self.model = OneClassSVM(nu=0.1, kernel='rbf', gamma=1e-3)
        self.arr = self.df[((self.df['User'] == user_num) & (df["Segment"].isin(range(50))))]
        self.y_train = self.arr.pop('Label')
        self.x_train = self.arr
        if user_num < 10:
            self.arr_test = self.df[((self.df['User'] == user_num) & (~(df["Segment"].isin(range(50)))))]
            self.y_test = self.arr_test.pop('Label')
            self.x_test = self.arr_test

    def optimize_parameters(self):
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['auto', 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                             'nu': [0.1, 0.2, 0.3, 0.5]}]
        # tuned_parameters = { 'novelty': [True], 'n_neighbors': range(1,21, 2),
        #                     'contamination': ['legacy', 0.1, 0.2, 0.3, 0.5]}
        # Split the dataset in two equal parts
        # tuned_parameters = { 'n_estimators': range(1,40,3), 'contamination': [0.1, 0.2]}
        self.train_models()
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            self.x_train, self.y_train, test_size=0.5, random_state=0)

        score = 'accuracy'
        print "# Tuning hyper-parameters for %s" % score
        clf = GridSearchCV(self.model, tuned_parameters, cv=10, scoring=score)
        clf.fit(X_train1, y_train1)

        print clf.best_params_

    def train_models(self):
        df = self.df.copy()
        other_user_df = pd.DataFrame([])
        result_df = pd.DataFrame([])
        normal_user_df = df[(df["User"] == self.user_num) & (df["Segment"].isin(range(50)))].copy()
        normal_user_df['Label'] = 0
        for other in range(40):
            if self.user_num != other:
                other_user_df = df[(df['User'] == other) & (df["Segment"].isin(range(50)))].copy()
                other_user_df['Label'] = 1
                train_df = normal_user_df.append(other_user_df)
                result_df = result_df.append(train_df)
        result_df = other_user_df
        y_train = result_df.pop('Label')
        x_train = result_df.drop(columns=['Segment', 'User'])
        return self.model.fit(self.x_train, self.y_train)

        '''
        other_user_df = df[(df['User'] == ((self.user_num + 1) % 40)) & (df["Segment"].isin(range(50)))].copy()
        other_user_df['Label'] = 1
        result_df = normal_user_df.append(other_user_df)
        self.y_train = result_df.pop('Label')
        self.x_train = result_df.drop(columns=['Segment', 'User', 'User_index', 'Segment_index'])
        if model is None:
            return self.model.fit(self.x_train, self.y_train)
        else:
            return model.fit(self.x_train, self.y_train)
        '''

    def predictLabels(self):
        self.train_models()

        # Save model
        filename = 'Final_Model.sav'
        pickle.dump(self.model, open(filename, 'wb'))

        # Load model and use it to make new predictions
        loaded_model = pickle.load(open(filename, 'rb'))

        # Load test dataset
        df = pd.read_csv(self.feature_select_output_file)
        arr = self.df[((self.df['User'] == self.user_num) & (~(self.df["Segment"].isin(range(50)))))]
        X = arr.drop(columns=['Label'])
        preds = self.model.predict(X)
        correct_preds = []
        """
        for pred in preds:
            if pred == -1:
                correct_preds.append(1)
            else:
                correct_preds.append(0)
        """
        print preds
        return preds
        print np.asarray(correct_preds)
        return np.asarray(correct_preds)