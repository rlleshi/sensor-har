import csv
import os.path as osp

import h5py
import matplotlib.pyplot as plt
import numpy as np

from rich.console import Console

plt.style.use('ggplot')
CONSOLE = Console()


class data_reader:
    ANN_FILE_PATH = 'data/annotations/zim-dance-10.txt'
    DATA_PATH = 'data/raw/zim/merged/'

    def __init__(self, train_test_files, use_columns, output_file_name, verbose):
        if not osp.exists(output_file_name):
            self.data, self.idToLabel = self.read_zim(train_test_files, use_columns, verbose)
            self.save_data(output_file_name)


    def save_data(self, output_file_name):
        with h5py.File(output_file_name, 'w') as f:
            for key in self.data:
                f.create_group(key)
                for field in self.data[key]:
                    f[key].create_dataset(field, data=self.data[key][field])


    def normalize(self, x):
        "Min-Max normalization"
        min_val, max_val = -1, 1
        return (x - min_val) / (max_val - min_val)


    @property
    def train(self):
        return self.data['train']


    @property
    def test(self):
        return self.data['test']


    def read_zim(self, train_test_files, use_columns, verbose):
        files = train_test_files
        with open(self.ANN_FILE_PATH, 'r') as ann:
            label_map = []
            for line in ann:
                (id, label) = line.split(' ', 1)
                label_map.append((id, label.strip()))

        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]
        if verbose:
            CONSOLE.print('\n =====   Label maps   =====\n', style='green')
            print(label_map)
            CONSOLE.print('\n =====   Label to Id   =====\n', style='green')
            print(labelToId)
            CONSOLE.print('\n =====   Id to Label   =====\n', style='green')
            print(idToLabel)

        cols = use_columns
        data = {dataset: self.read_zim_files(files[dataset], cols, labelToId)
                for dataset in ('train', 'test', 'validation')}
        return data, idToLabel

    def read_zim_files(self, filelist, cols, labelToId):
        data = []
        labels = []
        for _, filename in enumerate(filelist):
            # print('Reading file %d of %d' % (i + 1, len(filelist)))
            with open(osp.join(self.DATA_PATH, filename), 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    for ind in cols:
                        elem.append(line[ind])

                    data.append([float(x) / 10 for x in elem[1:]])
                    labels.append(labelToId[elem[0]])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}


def read_dataset(train_test_files, use_columns, output_file_name, verbose):
    CONSOLE.print('[Reading ZIM] ...', style='bold green')
    data_reader(train_test_files, use_columns, output_file_name, verbose)
    CONSOLE.print('[Reading ZIM] : DONE', style='bold green')
