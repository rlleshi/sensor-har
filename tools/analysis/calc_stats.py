import json
import numpy as np
from scipy import stats

from argparse import ArgumentParser
from rich.console import Console

CONSOLE = Console()


def parse_args():
    parser = ArgumentParser(
        prog='calculate sample statistics')
    parser.add_argument('sample', help='sample path')
    parser.add_argument(
        '--sensor-type',
        type=str,
        default='acc',
        choices=['both', 'acc', 'gyro'],
        help='type of sensor to plot')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.sensor_type == 'both':
        sensors = ['acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']
        balancer = 0
    elif args.sensor_type == 'acc':
        sensors = ['acc_x', 'acc_y', 'acc_z']
        balancer = 0
    else:
        sensors = ['gy_x', 'gy_y', 'gy_z']
        balancer = 3

    ind_to_sensor = {i+balancer: sensor for i, sensor in enumerate(sensors)}
    content = json.load(open(args.sample, 'r'))
    results = {k: [] for k in sensors}

    for row in content:
        for i in range(len(sensors)):
            results[ind_to_sensor[i+balancer]].append(row[i+balancer])

    for s in sensors:
        CONSOLE.print(f'Sensor: {s} | Mean: {np.mean(results[s])} \n',
                      f'groupped 25th percentile: {np.mean(np.percentile(results[s], [21, 22, 23, 24, 25, 26, 27]))}',
                      f'25th percentile: {np.percentile(results[s], 25)} \n',
                      f'Median: {np.median(results[s])}',
                      f'75th percentile: {np.percentile(results[s], 75)} \n',
                      f'Mode: {stats.mode([round(r, 2) for r in results[s]])}',
                      f'Mode: {stats.mode([round(r, 2) for r in results[s]])[0][0]}',
                      f'Std: {np.std(results[s])}', style='green')


if __name__ == '__main__':
    main()
