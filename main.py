import numpy as np

from DataProcessor import DataProcessor
from Vectorizer import Vectorizer
from FeatureSelector import FeatureSelector
import pandas as pd
from ClassificationModel import ClassificationModel
import csv

partial_labels_path= 'resources/partial_labels.csv'


"""
function that vectorizes all the users by segments, then reads the label of this segment in partial_labels.
writes the vectorization with the label of the segment to csv. 
"""

label_file = 'resources/partial_labels.csv'


def load_label_user(user_num):
    with open(label_file, 'rt') as f:
        rows = csv.reader(f, delimiter=',')
        next(rows, None)  # skip header
        list_rows = list(rows)
        return list_rows[user_num][1:]


def get_labels_array(user_num):
    input_list = load_label_user(user_num)
    label_list = []
    for i in range(len(input_list)):
        if input_list[i] == str(0) or input_list[i] == str(1):
            label_list.append(input_list[i])
        else:
            label_list.append(2) #2 means 'unknown!'
    return np.asarray(label_list)

def vectorize_all(n, type='ngram'):
    dp_list = [DataProcessor().load_raw_data_single_user_segments(user_num,num_of_segments=150) for user_num in range(40)]
    vectorizer = Vectorizer(ngram_count=n, type=type)
    pdfs = []
    for user_num in range(len(dp_list)):
        user_result = vectorizer.vectorize(dp_list[user_num], to_array=True)
        user_pdf = pd.DataFrame(user_result, columns=vectorizer.get_features())
        user_pdf['User'] = user_num
        user_pdf['Segment'] = np.arange(150)
        user_pdf['Label'] = get_labels_array(user_num)

        user_pdf.to_csv('outputs/Vectorizer/{}-{}-user{}.csv'.format(type, n, user_num))
        pdfs.append(user_pdf)
        del user_pdf
        print "Successfully vectorized user{} !".format(user_num)

    result_pdf = pd.concat(pdfs, ignore_index=True, axis=0, sort=True)
    result_pdf.to_csv('outputs/Vectorizer/all.csv')


def feature_select_all(df, n_features=250, n=2, type='ngram', write=True):
    fs = FeatureSelector()
    #for i in range(40):
    #    fs.select_most_common(i, n_features, n, type, write)
    result_pdf = fs.select_features(df, n_features=250, write=True)
    return result_pdf

def calc_stats_on_model(results, length):
    stats = [(results[0][i][0],
              (sum(results[n][i][1] for n in range(len(results)))/len(results)),
              (sum(results[n][i][2] for n in range(len(results)))/len(results)))
             for i in range(length)]
    return stats


if __name__ == "__main__":
    '''
    vectorize_all(2, 'ngram')
    result_pdf = pd.read_csv('outputs/Vectorizer/all.csv', dtype=pd.Int64Dtype(), na_values='')
    result_pdf.fillna(0, inplace=True)
    fs_result_df = FeatureSelector().select_features(result_pdf, n_features=250, write=True)
    '''
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

    for num in range(10, 40):
        modelsUsersArr[num].predictLabels(user_num=num, n=2, type='ngram')
    print 'Done'

'''
  #targets = [0] * len(data)
  #OneClassSVM().oneclass_svm_baseline(data, targets)
  #print 'Done'
'''