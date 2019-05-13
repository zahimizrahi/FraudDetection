import numpy as np

from DataProcessor import DataProcessor
from Vectorizer import Vectorizer
from FeatureSelector import FeatureSelector
import pandas as pd
from ClassificationModel import ClassificationModel
import csv

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

partial_labels_path= 'resources/partial_labels.csv'


"""
function that vectorizes all the users by segments, then reads the label of this segment in partial_labels.
writes the vectorization with the label of the segment to csv. 
"""

label_file = 'resources/partial_labels.csv'
sample_submission_file = 'resources/sample_submission.csv'
submission_file = 'final_submission.csv'
validation_file = 'validation.csv'


def calc_stats_on_model(results, length):
    stats = [(results[0][i][0],
              (sum(results[n][i][1] for n in range(len(results)))/len(results)),
              (sum(results[n][i][2] for n in range(len(results)))/len(results)))
             for i in range(length)]
    return stats

def validation(results, validation_set):
    grade = 0.0
    for i in range(len(results)):
        if results[i] == 1 and validation_set[i] == 1:
            grade += 9
        elif results[i] == 0 and validation_set[i] == 0:
            grade += 1
    print "\nGrade: " + str(grade/5400)

if __name__ == "__main__":
    """
    vectorize_all(2, 'ngram')
    result_pdf = pd.read_csv('outputs/Vectorizer/all.csv', dtype=pd.Int64Dtype(), na_values='')
    result_pdf.fillna(0, inplace=True)
    fs_result_df = FeatureSelector().select_features(result_pdf, n_features=250, write=True)
    

    
    results = []
    modelsUsersArr = []
    for num in range(40):
        print "******* User {} ********".format(num)
        classificationModel = ClassificationModel(user_num=num)
        modelsUsersArr.append(classificationModel)
        results.append(classificationModel.compare_models())
    stats = calc_stats_on_model(results, len(results[0]))
    stats.sort(key=lambda x: x[1], reverse=True)
    print stats
    """
    '''
    sample_df = pd.read_csv(sample_submission_file)
    result_df = sample_df
    for num in range(10, 40):
        print "******* User {} ********".format(num)
        classification_res = ClassificationModel(user_num=num).predictLabels()
        sample_df.loc[sample_df['id'].str.startswith('User{}_'.format(num)), 'label'] = classification_res
    print sample_df
    sample_df.shape
    sample_df.columns
    sample_df.to_csv(submission_file, index=False)
    print 'Done'
    '''

    #print FeatureSelector().select_features(write=True)
    #a = pd.Series(  DataProcessor().get_all_commands_series())
    #print a
    #commands = pd.Series(DataProcessor().get_all_commands_series())
    #print commands.keys()

    v = pd.read_csv(validation_file)
    validation_set = v['Label']
    classification_res = []
    for num in range(0, 10):
        print "******* User {} ********".format(num)
        classification_res.extend(ClassificationModel(user_num=num).predictLabels())
    validation(classification_res, validation_set)

    # Scatterplot Matrix
    df = pd.read_csv('outputs/FeatureSelector/all.csv')
    df = df.drop(columns=['Label', 'Segment', 'User', 'User_index', 'Segment_index'])
    scatter_matrix(df)
    plt.show()