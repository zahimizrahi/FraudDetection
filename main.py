from DataProcessor import DataProcessor
from Vectorizer import Vectorizer
from OneClassSVM import OneClassSVM
import os



if __name__ == "__main__":
  dp_list = [DataProcessor().load_raw_data_single_user_segments(user_num) for user_num in range(40)]
  #dp1 = DataProcessor().load_raw_data_single_user_segments(32)
  #dp2 = DataProcessor().load_raw_data_single_user_segments(33)
  vectorizer = Vectorizer(2, 'tfidf')
  data = vectorizer.vectorize(dp_list, to_array=True)
  targets = [0] * len(data)
  OneClassSVM().oneclass_svm_baseline(data, targets)
  print 'Done'