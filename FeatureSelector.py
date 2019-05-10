import pandas as pd
import csv
import numpy as np
from collections import Counter

class FeatureSelector:
    label_file = 'resources/partial_labels.csv'
    vectorize_dir_path = 'outputs/Vectorizer/'
    vectorize_output_path = 'outputs/Vectorizer/all.csv'
    feature_select_output_dir = 'outputs/FeatureSelector/'
    feature_select_output_file = 'outputs/FeatureSelector/all.csv'

    def select_most_common_all(self, df, n_features=250, write=False):
        num_users = df['User']
        segments = df['Segment']
        labels = df['Label']
        features_df = df.drop(columns=['User', 'Segment', 'Label', 'Unnamed: 0'])
        col_num = features_df.shape[1] - 1
        array = features_df.values
        X = array[:, 0:col_num]
        score = [0] * col_num
        for i in X:
            score += i  # summing all the scalar vectors

        common_features = dict(Counter(dict(zip(df.columns, score))).most_common(n_features))
        common_features_col = common_features.keys()

        result_df = df[common_features_col]

        result_df.loc[:, 'User'] = num_users
        result_df.loc[:, 'Segment'] = segments
        result_df.loc[:, 'Label'] = labels

        if write:
            result_df.to_csv(self.feature_select_output_file)
        return result_df

    def select_features(self, df, number_of_features=250, write=False):
        return self.select_most_common_all(df, n_features=number_of_features, write=write)



