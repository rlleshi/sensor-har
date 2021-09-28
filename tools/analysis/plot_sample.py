import json
import string

import pandas as pd
import seaborn as sns
import os.path as osp
import random as rd

from argparse import ArgumentParser
from rich.console import Console
from pathlib import Path


DATASET = 'zim'
CONSOLE = Console()


def gen_id(size=8):
    chars = string.ascii_uppercase + string.digits
    return ''.join(rd.choice(chars) for _ in range(size))


def clean(str):
    return str.replace('_', ' ').lower()


def parse_args():
    parser = ArgumentParser(
        prog='plot single sample')
    parser.add_argument('sample', help='sample path')
    parser.add_argument(
        '--sensor-type',
        type=str,
        default='both',
        choices=['both', 'acc', 'gyro'],
        help='type of sensor to plot')
    parser.add_argument(
        '--out-dir',
        type=str,
        default=f'results/samples/{DATASET}',
        help='out dir to save results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.sensor_type == 'both':
        ind = list(range(0, 6))
    elif args.sensor_type == 'acc':
        ind = list(range(0, 3))
    else:
        ind = list(range(3, 6))

    content = json.load(open(args.sample, 'r'))
    results = {k: [] for k in ['acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']}
    palette = ['Reds', 'Blues', 'Greys', 'Oranges', 'Purples', 'Greens']

    for row in content:
        results['acc_x'].append(row[0])
        results['acc_y'].append(row[1])
        results['acc_z'].append(row[2])
        results['gy_x'].append(row[3])
        results['gy_y'].append(row[4])
        results['gy_z'].append(row[5])

    sns.set(rc={'figure.figsize': (15, 13)})
    i = -1
    for k in results.keys():
        i += 1
        if i not in ind:
            continue

        df = pd.DataFrame({'Sample Index': list(range(0, len(results[k]))),
                    'Sensor Reading': results[k], 'Sensor': k})
        fig = sns.lineplot(x='Sample Index', y='Sensor Reading', data=df, hue='Sensor', palette=palette[i])
        fig.set_xlabel('Sample Index', fontsize=30)
        fig.set_ylabel('Sensor Reading', fontsize=20)

    output = fig.get_figure()
    movement, person = args.sample.split('/')[-1], args.sample.split('/')[-2]
    movement = clean(movement).split(' ')
    movement = '-'.join(movement[i] for i in range(0, 5))
    output.savefig(osp.join(args.out_dir,
        f'{movement}_{args.sensor_type}_{person}_{gen_id(3)}.svg'))

if __name__ == '__main__':
    main()
