import csv
import os.path as osp
import yaml
import json
import pandas as pd
import seaborn as sns

from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
from rich.console import Console


DATASET = 'zim'
CONSOLE = Console()

def parse_args():
    parser = ArgumentParser(prog='analysis of sample distribution')
    parser.add_argument(
        '--out-dir',
        type=str,
        default=f'results/sample_distribution/{DATASET}',
        help='out dir to save results')
    parser.add_argument(
        '--ann',
        type=str,
        default='data/annotations/zim-dance-10.txt',
        help='annotation file')
    args = parser.parse_args()
    return args


def save_result(out, result):
    cls = [k for k in result.keys()]
    val = [v for v in result.values()]
    tot = sum(val)
    val = list(map(lambda x: x/tot, val))

    # save json
    result['total'] = tot
    result_json = json.dumps(result, indent=4)
    f = open(osp.join(out, 'zim_classes_distribution.json'), 'w')
    print(result_json, file=f)
    f.close()

    # save plot
    df = pd.DataFrame({'Class': cls, 'Value': val})

    sns.set(rc={'figure.figsize': (15, 13)})
    fig = sns.barplot(x='Class', y='Value', data=df)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=15)
    fig.axes.set_title('Sample Distribution / Class ', fontsize=40)
    fig.set_xlabel('Class', fontsize=30)
    fig.set_ylabel('Value', fontsize=20)
    output = fig.get_figure()
    output.savefig(osp.join(out, 'zim_classes_distribution.svg'))


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    config_file = open('configs/data.yaml', 'r')
    data_config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = data_config[DATASET]

    number_to_label = {}
    with open(args.ann) as ann:
        for line in ann:
            (val, key) = line.split(' ', 1)
            number_to_label[val] = key.strip()

    result = {key: 0 for key in number_to_label.keys()}

    for data in tqdm(config['train_files'] + \
                config['validation_files'] + \
                config['test_files']):
        path = osp.join(data_config['data_dir']['raw'],
                        DATASET, 'merged', data)
        # CONSOLE.print(f'Examining {path}...', style='bold yellow')

        with open(path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            for row in csv_reader:
                result[row[0]] +=1

    result = {number_to_label[k]: v for k, v in result.items()}
    save_result(args.out_dir, result)


if __name__ == '__main__':
    main()
