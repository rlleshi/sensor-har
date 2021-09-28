import json
import yaml
import os
import os.path as osp

import tensorflow as tf
import numpy as np

from argparse import ArgumentParser
from pathlib import Path
from sklearn.metrics import accuracy_score
from rich.console import Console

CONSOLE = Console()

def segment_window_all(x_train, y_train, window_size, n_sensor_val):
    window_segments = np.zeros((len(x_train), window_size, n_sensor_val))
    labels = np.zeros((len(y_train),))

    total_len = len(x_train)

    for i in range(total_len):
        end = i + window_size

        if end > total_len:
            pad_len = end - total_len
            window_segments[i] = x_train[i - pad_len:end]
            labels[i] = y_train[total_len - 1]
        else:
            window_segments[i] = x_train[i:end]
            labels[i] = y_train[end - 1]

    return window_segments, labels


def clean(str):
    return str.replace('_', ' ').lower()


def parse_args():
    parser = ArgumentParser(prog='get prediction for single sample')
    parser.add_argument('path',
        type=str,
        help='path to sample or dir')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='saved_model/zim',
        help='model dir')
    parser.add_argument(
        '--ann',
        type=str,
        default='data/annotations/zim-dance-10.txt',
        help='annotation file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='data/raw/zim_dance/single_test',
        help='out dir to save processed sample')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    config_file = open('configs/data.yaml', 'r')
    data_config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = data_config['zim']
    model_file = open('configs/model.yaml', 'r')
    model_config = yaml.load(model_file, Loader=yaml.FullLoader)
    model = tf.keras.models.load_model(args.model_dir)

    # data pre-processing
    label_to_number = {}
    with open(args.ann, 'r') as ann:
        for line in ann:
            (val, key) = line.split(' ', 1)
            label_to_number[key.strip()] = int(val)

    to_process = []
    if osp.isdir(args.path):
        to_process += os.listdir(args.path)
    else:
        to_process.append(args.path.split('/')[-1])
        args.path = '/'.join(p for p in args.path.split('/')[:-1])

    # CONSOLE.print(to_process, style='green')
    # CONSOLE.print(args.path, style='green')

    for sample in to_process:
        label = clean(sample.split('2021')[0])
        activity_id = label_to_number.get(label.strip(), None)
        x_test = []
        y_test = []

        content = open(osp.join(args.path, sample), 'r')
        try:
            content = json.load(content)
        except:
            continue

        for row in content:
            result = [activity_id]
            result.extend([x for x in row])
            x_test.append([float(x) / 10 for x in result[1:]])
            y_test.append(result[0])

        n_sensor_val = len(config['feature_columns']) - 1
        # replace any nan with mean
        x_test = np.where(
            np.isnan(x_test),
            np.ma.array(x_test, mask=np.isnan(x_test)).mean(axis=0), x_test)

        # window
        test_x, test_y = segment_window_all(x_test, y_test, config['window_size'], n_sensor_val)
        test_y = tf.keras.utils.to_categorical(test_y)

        # predict
        pred = model.predict(
            test_x,
            batch_size=model_config['zim']['batch_size'],
            verbose=1)
        acc = accuracy_score(
            np.argmax(test_y, axis=1),
            np.argmax(pred, axis=1),
            normalize=True)
        print(sample)
        CONSOLE.print(f'The model is {round(100*acc, 2)}% confident '
                    f'that this sample is {label}', style='green')


if __name__ == '__main__':
    main()
