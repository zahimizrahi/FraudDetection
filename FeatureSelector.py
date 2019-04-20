import pandas as pd
import csv
import numpy as np
from collections import Counter

class FeatureSelector:
    label_file = 'resources/partial_labels.csv'
    vectorize_dir_path = 'outputs/Vectorizer/'
    feature_select_output_dir = 'outputs/FeatureSelector/'

    def load_data_user(self, user_num, n=2, type='ngram'):
        return pd.read_csv(self.vectorize_dir_path+'{}-{}-user{}.csv'.format(type, n, user_num))

    def load_label_user(self, user_num):
        with open(self.label_file, 'rt') as f:
            rows = csv.reader(f,delimiter=',')
            next(rows,None) #skip header
            list_rows = list(rows)
            return list_rows[user_num][1:]

    def get_labels_array(self, user_num):
        input_list = self.load_label_user(user_num)
        label_list=[]
        for i in range(len(input_list)):
            if input_list[i] == str(0) or input_list[i] == str(1):
                label_list.append(str(input_list[i]))
            else:
                label_list.append('None')
        return np.asarray(label_list)

    def select_most_common(self, user_num, n_features = 100, n=2, type='ngram', write=False):
        df = self.load_data_user(user_num, n, type)
        user_list = self.load_label_user(user_num)
        col_num = df.shape[1] - 1
        array = df.values
        X = array[:, 0:col_num] # count for each feature
        Y = self.get_labels_array(user_num)

        score = [0] * col_num
        for i in X:
            score+= i # summing all the scalar vectors

        common_features = dict(Counter(dict(zip(df.columns, score))).most_common(n_features))
        common_features_col = common_features.keys()

        result_df = df[common_features_col]
        result_df.loc[:, 'Label'] = Y

        if write:
            result_df.to_csv('outputs/FeatureSelector/{}-{}-user{}.csv'.format(type, n, user_num))
        return result_df

    def select_features(self, user_num, number_of_features=100, n=2, type='ngram', write=False):
        return FeatureSelector().select_most_common(0, number_of_features=100, n=n, type=type, write=write)

