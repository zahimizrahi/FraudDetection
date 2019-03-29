from DataProcessor import *
import os



if __name__ == "__main__":
    dp = DataProcessor().load_raw_data_single_user_segments(32)
    print len(dp)
