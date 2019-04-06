import os


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
            commands = [command.rstrip('\n') for command in user_file.readlines()]
        return ' '.join(commands)

    """
    load the raw data from resources of a single segment for a single user.
    the result will be flat list with all the commands in this segment for the user. 
    """
    def load_raw_data_single_segment(self, num_user, num_segment):
        path = os.path.join(self.raw_data_dir_path, self.raw_data_filename + str(num_user))
        with open(path) as user_file:
            commands = [command.rstrip('\n') for command in user_file.readlines()]
        start = self.sample_size * (num_segment)
        end =  self.sample_size * (num_segment + 1)
        return ' '.join(commands[start:end])

    """
    load the raw data from resources of a single user divisioned to segments. 
    the result will be list when element i will be the commands of the user in segment i. 
    """
    def load_raw_data_single_user_segments(self, num_user, num_of_segments=50):
        raw_data_list = [self.load_raw_data_single_segment(num_user, segment) for segment in range(0, num_of_segments)]
        return ' '.join(raw_data_list)