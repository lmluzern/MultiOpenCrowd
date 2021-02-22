# Multiclass Classification of Open-ended Answers
This repo belongs to the master thesis Multiclass Classification of Open-ended Answers. Goal is to infer the truth from crowdsourced open-ended answers in a multiclass setting. For this purpose, we extend existing methods and develop a new one. Below are the instructions for reproducing the experiments presented in the thesis.

## Overview
   * `src/non_feature_based/opencrowd_gibbs` Implementation of our non-feature-based method called OpenCrowd with Gibbs Sampling.
   * `src/non_feature_based/baselines` Non-feature-based baseline methods (CATD,D&S,GLAD,LFC,MV). Implementations are taken from https://github.com/TsinghuaDatabaseGroup/CrowdTI and adapted where necessary.
   * `src/feature_based/multiclass_opencrowd` Is our extension of OpenCrowd so that it covers the multiclass case. The original implementation can be found here: https://github.com/eXascaleInfolab/OpenCrowd.
   * `scr/feature_based/baselines/bccwords_mlp` Feature-based baseline method (BCCWords). The implementation is taken from https://github.com/UKPLab/arxiv2018-bayesian-ensembles (Apache-2.0 License). With the aim of fair comparison, we adapted the implementation. In short, we integrated the same classifier that we use in Multiclass OpenCrowd.

Note that FaitCrowd, another feature-based baseline method, is not publicly available, so it is not included in this repo.

## Installation
Scripts were successfully tested with Python 3.6.9.

### Setup virtual environment
The following statement will set up the virtual environment and install the required libraries.

``` bash 
$ source installation.sh
```

## Usage
To reproduce non-feature-based experiments, run the following script. You must pass the dataset as argument [influencer,sentiment_sparse,sentiment].
``` bash
python src/exp_non_feature_based.py influencer
```

tbd...
To reproduce Multiclass OpenCrowd experiments, use src/experiments.py. There are some examples of run_experiment() provided.
To reproduce baseline experiments (LFC, etc.), use baseline methods/experiments.py. There are some examples of evaluate_supervision_rate() provided.
You can use the var_em.py script provided in src to apply Multiclass OpenCrowd on the provided data set:
``` bash
python3 src/var_em.py
```

## Citation
Please cite the following paper when using OpenCrowd:
``` bash
@inproceedings{arous2020www,
  title = {OpenCrowd: A Human-AI Collaborative Approach for Finding Social Influencers via Open-Ended Answers Aggregation},
  author = {Arous, Ines and Yang, Jie and Khayati, Mourad and Cudr{\'e}-Mauroux, Philippe},
  booktitle = {Proceedings of the Web Conference (WWW 2020)},
  year = {2020},
  address = {Taipei, Taiwan}
}
```