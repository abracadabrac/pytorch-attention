
import json
import os


class Vars:
    def __init__(self):
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

        file = open('../../vars.json', 'r')
        v = json.load(file)

        self.images_train_dir = v['images_train_dir']
        self.images_valid_dir = v['images_valid_dir']
        self.images_test_dir = v['images_test_dir']

        self.labels_train_txt = v['labels_train_txt']
        self.labels_valid_txt = v['labels_valid_txt']
        self.labels_test_txt = v['labels_test_txt']

        self.encoding_dict = v['encoding_dict']
        self.decoding_dict = v['decoding_dict']

        self.experiments_folder = v['experiments_folder']


if __name__ == '__main__':
    
    vrs = Vars()
