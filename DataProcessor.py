import os
import pandas as pd

class DataProcessor:
    raw_data_dir_path = 'resources/FraudedRawData/'
    raw_data_filename = 'User'
    num_of_segments = 150
    num_of_users = 40
    num_of_benign_segments = 50
    sample_size = 100

    def __init__(self):
        return

    """
    load the raw data from resources of a single user sequentally (without division to segments).
    the result will be flat list with all the commands.  
    """
    def load_raw_data_single_user_all(self, num_user):
        path = os.path.join(self.raw_data_dir_path, self.raw_data_filename + str(num_user))
        with open(path) as user_file:
            commands = [command.rstrip('\n').replace('-','').replace('.','') for command in user_file.readlines()]
        return ' '.join(commands)


    def split_raw_data_to_segments_user_all(self, num_user, num_of_segments=150):
        path = os.path.join(self.raw_data_dir_path, self.raw_data_filename + str(num_user))
        with open(path,'rb') as user_file:
            lines = [r.rstrip('\n') for r in user_file.readlines()]

        user_segments = []
        for i in range(num_of_segments):
            start = self.sample_size * i
            end = (self.sample_size * (i+1))
            user_segments.append(lines[start:end])
        return user_segments

    """
    load the raw data from resources of a single segment for a single user.
    the result will be flat list with all the commands in this segment for the user. 
    """
    def load_raw_data_single_segment(self, num_user, num_segment):
        path = os.path.join(self.raw_data_dir_path, self.raw_data_filename + str(num_user))
        with open(path) as user_file:
            commands = [command.rstrip('\n').replace('-','').replace('.','') for command in user_file.readlines()]
        start = self.sample_size * (num_segment)
        end =  self.sample_size * (num_segment + 1)
        return ' '.join(commands[start:end])

    """
    load the raw data from resources of a single user divisioned to segments. 
    the result will be list when element i will be the commands of the user in segment i. 
    """
    def load_raw_data_single_user_segments(self, num_user, num_of_segments=50):
        raw_data_list = [self.load_raw_data_single_segment(num_user, segment) for segment in range(0, num_of_segments)]
        return raw_data_list

    def get_all_commands_series(self):
        commands = {}
        for user in os.listdir(self.raw_data_dir_path):
            if user.startswith('User'):
                user_num = int(user.split('User')[1])
                commands[user_num] = self.split_raw_data_to_segments_user_all(user_num, num_of_segments=150)
        return commands