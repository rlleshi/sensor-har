# Evaluating Dance Movements Through Sensor-Based Human Action Recognition

The aim of this project is to evaluate student's dance movements. Students will be holding their iPhone during their performance and generated accelerometer and gyroscope data will be feed to a sensor-based HAR model to check the quality of the dance.
# Dance Movements

- English Valse
    - Chasse bg
    - Chasse fw
    - Natural turn bw 1-3
    - Natural turn fw 1-3
    - Natural turn starting bw 1-6
    - Natural turn starting fw 1-6
- Tango
    - 2 walking steps bw
    - 2 walking steps fw
    - Rock turn starting bw
    - Rock turn starting fw

# Technical
## Installation

To install the dependencies in `python3` environment, run:

```shell
pip install -r requirements.txt
```


## Pretrained Models

The `saved_model` directory contains pretrained models for various feature sets (apart from sensor data). It was found that feeding the length of each sample, as well as basic statistics such as mean and std. significatly boosted the accuracy. Three different such models could be found here.
These models can be used directly for inference and performance evaluation as described in the following section.

## Training and Evaluation

Python script `main.py` will be used for model training, inference and performance evaluation. The arguments for this
script are as follows:

    -h, --help         show this help message and exit
    --train            Training Mode
    --test             (Testing / Evaluation) Mode
    --epochs EPOCHS    Number of Epochs for Training
    --dataset DATASET  Name of Dataset for Model Training or Inference

For example, in order to train model for `75` epochs on `PAMAP2` dataset and evaluate model performance, run the
following command:

```shell
TF_CPP_MIN_LOG_LEVEL=3 python main.py --train --test --epochs 75 --dataset pamap2
```

If the pretrained weights are stored in `saved_model` directory and to infer with that, run the following command:

```shell
TF_CPP_MIN_LOG_LEVEL=3 python main.py --test --dataset pamap2
```

Make sure to tune the `feature_columes` in `config/data.yaml` accordingly

### References

[24th European Conference on Artificial Intelligence, ECAI 2020](https://digital.ecai2020.eu/)
by [Saif Mahmud](https://saif-mahmud.github.io/) and M. Tanjid Hasan Tonmoy et al.

[ [arXiV](https://arxiv.org/abs/2003.09018) ] [ [IOS Press](https://ebooks.iospress.nl/publication/55031) ]
