import pandas as pd
import csv
import numpy as np
from collections import Counter
from collections import defaultdict
from DataProcessor import DataProcessor
from Vectorizer import Vectorizer

class FeatureSelector:
    label_file = 'resources/partial_labels.csv'
    vectorize_dir_path = 'outputs/Vectorizer/'
    vectorize_output_path = 'outputs/Vectorizer/all.csv'
    feature_select_output_dir = 'outputs/FeatureSelector/'
    feature_select_output_file = 'outputs/FeatureSelector/all.csv'


# added michal features

    def command_avg_length(self, segments_list):
        segments_avg_len = []
        for seg in segments_list:
            counter = 0
            commands_list = seg.split(" ")
            for cmd in commands_list:
                counter += len(cmd)
            segments_avg_len.append(counter / len(segments_list))
        return segments_avg_len

    def diff_commands_in_seg(self, segment_list):
        diff_commands_in_segment = []
        for seg in segment_list:
            unique_list = []
            commands_list = seg.split(" ")
            for cmd in commands_list:
                if cmd not in unique_list:
                    unique_list.append(cmd)
            diff_commands_in_segment.append(len(unique_list))
        return diff_commands_in_segment

    def num_of_sequences(self, segment_list):
        num_of_seq_in_seg = []
        for seg in segment_list:
            counter = 0
            commands_list = seg.split(" ")
            for i in range(len(commands_list)):
                if i + 1 < len(commands_list):
                    if commands_list[i] == commands_list[i + 1]:
                        counter += 1
            num_of_seq_in_seg.append(counter)
        return num_of_seq_in_seg

    def avg_len_of_sequences(self, segment_list):
        avg_len_of_seq = [0] * 150
        for seg in segment_list:
            counter = 0
            sum_seq_len = 0
            num_of_seq = 0
            commands_list = seg.split(" ")
            for i in range(len(commands_list)):
                if i + 1 < len(commands_list):
                    if commands_list[i] == commands_list[i + 1]:
                        counter += 1
                    elif counter > 0:
                        sum_seq_len += counter
                        num_of_seq += 1
                        counter = 0
            if num_of_seq > 0:
                avg_len_of_seq.append(sum_seq_len / num_of_seq)
        return avg_len_of_seq

# for zahi features

    def load_label_user(self,user_num):
        with open(self.label_file, 'rt') as f:
            rows = csv.reader(f, delimiter=',')
            next(rows, None)  # skip header
            list_rows = list(rows)
            return list_rows[user_num][1:]

    def get_list_of_sublist (self, lst, min_len, max_len):
        res = []
        for curr_len in range(min_len, max_len):
            for start in range( len(lst) - curr_len + 1):
                res.append(lst[start:(start + curr_len)])
        return res


    def get_labels_array(self,user_num):
        input_list = self.load_label_user(user_num)
        label_list = []
        for i in range(len(input_list)):
            if input_list[i] == str(0) or input_list[i] == str(1):
                label_list.append(input_list[i])
            else:
                label_list.append(2)  # 2 means 'unknown!'
        return np.asarray(label_list)

    def get_partial_labels(self):
        return pd.read_csv('resources/partial_labels.csv')

    def vectorize_all(self, n, type='ngram'):
        dp_list = [DataProcessor().load_raw_data_single_user_segments(user_num, num_of_segments=150) for user_num in
                   range(40)]
        vectorizer = Vectorizer(ngram_count=n, type=type)
        pdfs = []
        for user_num in range(len(dp_list)):
            user_result = vectorizer.vectorize(dp_list[user_num], to_array=True)
            user_pdf = pd.DataFrame(user_result, columns=vectorizer.get_features())
            user_pdf['User'] = user_num
            user_pdf['Segment'] = np.arange(150)
            user_pdf['Label'] = self.get_labels_array(user_num)

            user_pdf.to_csv('outputs/Vectorizer/{}-{}-user{}.csv'.format(type, n, user_num))
            pdfs.append(user_pdf)
            del user_pdf
            print "Successfully vectorized user{} !".format(user_num)

        result_pdf = pd.concat(pdfs, ignore_index=True, axis=0, sort=True)
        result_pdf.to_csv('outputs/Vectorizer/all-{}-{}.csv'.format(n, type))

    def select_most_common_all(self, df, n_features=250):
        num_users = df.pop('User')
        segments = df.pop('Segment')
        labels = df.pop('Label')
        features_df = df
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

        return result_df


    def select_features(self, write=True):


    # first feature: TOP 40 common words
    # top-40 most common commands at all (We will use 1 gram, but it can work also with 2/3 gram)
    # self.vectorize_all(n, type)

    # 1st feature
        one_ngram_df = pd.read_csv('outputs/Vectorizer/all-{}-{}.csv'.format(1,'ngram'))
        df = self.select_most_common_all(one_ngram_df,
                                         n_features=40)
        # first feature
        df.fillna(0, inplace=True)
        df.loc[:,'User_index'] = df['User']
        df.loc[:, 'Segment_index'] = df['Segment']
        df.set_index(['User_index', 'Segment_index'], inplace=True)
        print 'Finished 1 gram!'
    # 2nd feature: TOP 40 2-gram - NOT INCLUDED RIGHT NOW
    # top-40 most common 2-gram  sequences
    # self.vectorize_all(n, type)
     #   two_ngram_df = pd.read_csv('outputs/Vectorizer/all-{}-{}.csv'.format(2,'ngram'))
     #   df2 = self.select_most_common_all(two_ngram_df, n_features=40)
     #   df2.fillna(0, inplace=True)

      #  df.join(df2, how='left', on=['User', 'Segment', 'Label'])

    # third feature: NEW USED COMMANDS
    # number of commands that didn't appear in the first 50 segments, but appeared in the given chunk
    # (indication of a commands that are used recently now)
    # the feature will be 0 for the first 50 segments

    # 4th feature: count of fake commands
    # counter of commands that are used by malicious.
    # feature of unique commands that are used only as fake commands in the given segment.
    # feature of unique commands that are used only as benign commands in the given segment.
    # feature of number of commands from all the fake commands in the trainset for given segement.
    # feature of number of commands from all the benign commands in the trainset for given segement - NOT INCLUDED.


    # 5th feature: repeated sequence of commands
    # number of different repeated sequence of commands that appeared at least 4 times (for each lengths)
    # why? because legitimate user won't use sequence of commands repeatedly.

    # preparations for 3rd feature
        commands = pd.Series(DataProcessor().get_all_commands_series())
        print commands.keys()
        partial_labels = self.get_partial_labels()

        distinct_first_50_commands = set()
        for user_num in commands.keys():
            for segment in commands[user_num][:50]:
                for command in segment:
                    distinct_first_50_commands.add(command)

        print 'Finished distinct_first_50_commands!'
    # preparation for 4th feature
        malicious_commands = defaultdict(list)
        for i in range(50, 150):
            col_index = str(100 * i ) + '-' + str(100 * (i+1))
            for num_user in range(10):
                if partial_labels[col_index][num_user] == 1:
                    malicious_commands[num_user].extend(commands[num_user][i])

        malicious_commands_of_train_users_set = set()
        benign_commands_of_train_users_set = set()

        for num_user in range(10):
            malicious_commands_of_train_users_set = \
                malicious_commands_of_train_users_set.union(set(malicious_commands[num_user]))
            user = pd.Series(commands[user_num])
            for segment in user[:50]:
                benign_commands_of_train_users_set = benign_commands_of_train_users_set.union(set(segment))

        commands_used_only_by_malicious_train = \
            malicious_commands_of_train_users_set - benign_commands_of_train_users_set
        commands_used_only_by_benign_train = benign_commands_of_train_users_set - malicious_commands_of_train_users_set

        print 'Finished preparing sets of benign and malicious!'

        dp_list = [DataProcessor().load_raw_data_single_user_segments(user_num, num_of_segments=150) for user_num in
                   range(40)]

        user_cmd_avg_len = [self.command_avg_length(dp_list[user_num]) for user_num in range(40)]
        user_diff_cmd = [self.diff_commands_in_seg(dp_list[user_num]) for user_num in range(40)]
        user_num_of_seq = [self.num_of_sequences(dp_list[user_num]) for user_num in range(40)]

        print 'Finished preparing features of michal!'
    ### adding the additional features
        for user_num in commands.keys():
            for num_segment, segment in enumerate(commands[user_num]):

                # 3rd feature
                df.loc[(user_num, num_segment), 'NewUsedCommands'] = \
                    len(set(segment) - distinct_first_50_commands)

                # 4th feature
                df.loc[(user_num, num_segment), 'UniqueMaliciousCommands'] = \
                    len( set(segment) & commands_used_only_by_malicious_train)
                df.loc[(user_num, num_segment), 'UniqueMaliciousCommands'] = \
                    len(set(segment) & commands_used_only_by_benign_train)
                df.loc[(user_num, num_segment), 'MaliciousCommandsCount'] = \
                    len(set(segment) & malicious_commands_of_train_users_set)
                # df.loc[(user_num, num_segment), 'BenignCommandsCount'] = \
                #    len(set(segment) & benign_commands_of_train_users_set)

                # 5th feature
                min_len = 2
                max_len = 10
                minimum_seq_count = 4
                count_dict = {c: 0 for c in range(min_len, max_len)}
                lst = segment

                for sub in self.get_list_of_sublist(lst, min_len, max_len):
                    sub_list = list(sub)

                    counts = [1 if lst[i: (i+len(sub_list))] == sub_list else 0 for i in
                              range(len(segment) - len(sub_list))]

                    # we need to slice the slot in the list mapped by the length of the seq to avoid overlapping seqs.
                    count_sum = sum(1 for i in range(0, len(counts), len(sub_list))
                                    if (sum(counts[i:i + len(sub_list)]) > 0))

                    if count_sum > minimum_seq_count:
                        count_dict[len(sub)] += 1

                for count_key, count_val in count_dict.items():
                    df.loc[ (user_num, num_segment), 'Seq_of_commands_repeated_{}'.format(count_key)] = count_val


                # added michal features

                df.loc[(user_num, num_segment), 'Num_of_sequences'] = user_num_of_seq[user_num][num_segment]
                df.loc[(user_num, num_segment), 'Diff_commands'] = user_diff_cmd[user_num][num_segment]
                df.loc[(user_num, num_segment), 'Avg_commands_length'] = user_cmd_avg_len[user_num][num_segment]

                print 'Done loop: User {}, Segment {} ...'.format(user_num, num_segment)

        print 'Finished loop!'
        df.fillna(0, inplace=True)

        # remove overlapping counts
        df.loc[:, 'Seq_of_commands_repeated_2'] =\
            df['Seq_of_commands_repeated_2'] - df['Seq_of_commands_repeated_3']
        df.loc[:, 'Seq_of_commands_repeated_3'] = \
            df.loc[:, 'Seq_of_commands_repeated_3'] - df['Seq_of_commands_repeated_4']
        df.loc[:, 'Seq_of_commands_repeated_4'] = \
            df['Seq_of_commands_repeated_4'] - df['Seq_of_commands_repeated_5']
        df.loc[:, 'Seq_of_commands_repeated_5'] = \
            df['Seq_of_commands_repeated_5'] - df['Seq_of_commands_repeated_6']
        df.loc[:, 'Seq_of_commands_repeated_6'] = \
            df['Seq_of_commands_repeated_6'] - df['Seq_of_commands_repeated_7']
        df.loc[:, 'Seq_of_commands_repeated_7'] = \
            df['Seq_of_commands_repeated_7'] - df['Seq_of_commands_repeated_8']
        df.loc[:, 'Seq_of_commands_repeated_8'] = \
            df['Seq_of_commands_repeated_8'] - df['Seq_of_commands_repeated_9']
        del df['Seq_of_commands_repeated_9']
        del df['Seq_of_commands_repeated_8']
        del df['Seq_of_commands_repeated_7']

        del df['Seq_of_commands_repeated_5']

        del df['Seq_of_commands_repeated_2']

        # added michal features
        dp_list = [DataProcessor().load_raw_data_single_user_segments(user_num, num_of_segments=150) for user_num in
               range(40)]

        print 'Before write...'
        if write:
            df.to_csv(self.feature_select_output_file)
        return df
