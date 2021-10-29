import os
import argparse
import warnings
import yaml
import tensorflow as tf

from utils.data import get_data
from utils.result import generate_result
from utils.test import test_model
from utils.train import train_model
from rich.console import Console

CONSOLE = Console()
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser(prog='self-attention-har')
    parser.add_argument(
        '--train',
        action='store_true',
        default=False,
        help='Training Mode')
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help='Testing Mode')
    parser.add_argument(
        '--epochs',
        default=150,
        type=int,
        help='Number of Epochs for Training')
    parser.add_argument(
        '--dataset',
        default='zim',
        type=str,
        help='Name of Dataset for Model Training')
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='whether to use GPU or not')
    parser.add_argument(
        '--save',
        action='store_true',
        help='whether to save the model or not')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not args.use_gpu:
        CONSOLE.print('Not using GPU', style='bold yellow')
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model_config_file = open('configs/model.yaml', mode='r')
    model_cfg = yaml.load(model_config_file, Loader=yaml.FullLoader)
    train_x, train_y, val_x, val_y, test_x, test_y = get_data(dataset=args.dataset)

    if args.train:
        CONSOLE.print('\n[MODEL TRAINING]', style='bold green')
        train_model(dataset=args.dataset,
                    model_config=model_cfg,
                    train_x=train_x, train_y=train_y,
                    val_x=val_x, val_y=val_y,
                    epochs=args.epochs, save_model=args.save)

    if args.test:
        CONSOLE.print('\n[MODEL INFERENCE]', style='bold green')
        pred = test_model(dataset=args.dataset, model_config=model_cfg,
                          test_x=test_x)
        generate_result(dataset=args.dataset, ground_truth=test_y,
                        prediction=pred)


if __name__ == '__main__':
    main()
