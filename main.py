from DataProcessor import DataProcessor
from Vectorizer import Vectorizer
from FeatureSelector import FeatureSelector
import pandas as pd
from ClassificationAlgo import ClassificationAlgo
import csv

partial_labels_path= 'resources/partial_labels.csv'


"""
function that vectorizes all the users by segments, then reads the label of this segment in partial_labels.
writes the vectorization with the label of the segment to csv. 
"""


def vectorize_all(n, type='ngram'):
  dp_list = [DataProcessor().load_raw_data_single_user_segments(user_num,num_of_segments=150) for user_num in range(40)]
  vectorizer = Vectorizer(ngram_count=n, type=type)
  data = pd.DataFrame()
  for user_num in range(len(dp_list)):
      user_result = vectorizer.vectorize(dp_list[user_num], to_array=True)
      user_pdf = pd.DataFrame(user_result, columns=vectorizer.get_features())
      user_pdf.to_csv('outputs/Vectorizer/{}-{}-user{}.csv'.format(type, n, user_num))


def feature_select_all(n_features=200, n=2, type='ngram', write=True):
    fs = FeatureSelector()
    for i in range(40):
        fs.select_most_common(i, n_features, n, type, write)
    return


if __name__ == "__main__":
  vectorize_all(2, 'ngram')
  feature_select_all(100, 2, 'ngram', True)
  # ClassificationAlgo(user_num=0).logisticRegression()
  # ClassificationAlgo(user_num=0).decisionTree()
  # ClassificationAlgo(user_num=0).oneclassSvm()
  ClassificationAlgo(user_num=0).compare_models()

  print 'Done'



  #targets = [0] * len(data)
  #OneClassSVM().oneclass_svm_baseline(data, targets)
  #print 'Done'