import csv
import json
import os
import os.path as osp

from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
from rich.console import Console

LABEL_TO_NUMBER = {}
CONSOLE = Console()


def clean(str):
    return str.replace('_', ' ').lower()


def parse_args():
    parser = ArgumentParser(
        prog='merge single samples (all classes) of one into 1 data file')
    parser.add_argument('in_dir', type=str, help='directory of samples')
    parser.add_argument(
        '--out',
        type=str,
        default='data/raw/zim_dance/merged/subjectXY.dat',
        help='resulting file')
    parser.add_argument(
        '--ann',
        type=str,
        default='data/annotations/zim-dance-10.txt',
        help='annotation file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    Path(args.out.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
    global LABEL_TO_NUMBER
    with open(args.ann, 'r') as ann:
        for line in ann:
            (val, key) = line.split(' ', 1)
            LABEL_TO_NUMBER[key.strip()] = int(val)

    with open(args.out, mode='w') as out:
        writer = csv.writer(out, delimiter=' ')

        for file in tqdm(os.listdir(args.in_dir)):
            label = clean(file.split('2021')[0])
            activity_id = LABEL_TO_NUMBER.get(label.strip(), None)
            content = open(osp.join(args.in_dir, file), 'r')
            content = json.load(content)

            CONSOLE.print(len(content), style='green')
            for row in content:
                result = [activity_id]
                result.extend([x for x in row])
                writer.writerow(result)


if __name__ == '__main__':
    main()
