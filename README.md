# Multiclass Classification of Open-ended Answers
This repo belongs to the master thesis Multiclass Classification of Open-ended Answers. Goal is to infer the truth from crowdsourced open-ended answers in a multiclass setting. For this purpose, we extend existing methods and develop a new one. Below are the instructions for reproducing the experiments presented in the thesis.

## Overview
   * `src/non_feature_based/opencrowd_gibbs` Implementation of our non-feature-based method called OpenCrowd with Gibbs Sampling.
   * `src/non_feature_based/baselines` Non-feature-based baseline methods (CATD,D&S,GLAD,LFC,MV). Implementations are taken from https://github.com/TsinghuaDatabaseGroup/CrowdTI and adapted where necessary.
   * `src/feature_based/multiclass_opencrowd` Is our extension of OpenCrowd so that it covers the multiclass case. The original implementation can be found here: https://github.com/eXascaleInfolab/OpenCrowd.
   * `scr/feature_based/baselines/bccwords_mlp` Feature-based baseline method (BCCWords). The implementation is taken from https://github.com/UKPLab/arxiv2018-bayesian-ensembles (Apache-2.0 License). With the aim of fair comparison, we adapted the implementation. In short, we integrated the same classifier that we use in Multiclass OpenCrowd.

Note that FaitCrowd, another feature-based baseline method, is not publicly available, so it is not included in this repo.

## Installation (MacOS/Unix)
Scripts were successfully tested with Python 3.6.9.

### Setup virtual environment
The following statement will set up the virtual environment and install the required libraries.

``` bash 
$ source installation.sh
```

## Reproducing Experiments
The resulting plot (.png) and the raw data (.csv) are exported to output/. 

### Non-feature-based experiment
You must pass the dataset as argument [influencer,sentiment_sparse,sentiment]. To reproduce the non-feature-based experiments with varying supervision rate, run the following script (example on Influencer dataset).
``` bash
python src/exp_non_feature_based.py influencer
```

### Feature-based experiment
You must pass the dataset as argument [influencer,sentiment_sparse,sentiment]. To reproduce the feature-based experiments with varying supervision rate, run the following script (example on Influencer dataset).
``` bash
python src/exp_feature_based.py influencer
```
