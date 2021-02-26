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
First, navigate to the project folder.
``` bash
$ cd MultiOpenCrowd/
```
The following statement will set up the virtual environment and install the required libraries. Pip will be upgraded.
``` bash
$ source installation.sh
```

## Reproducing Experiments
To reproduce the experiments we report in sections 4.2, 4.3, and 4.4 of the thesis, you can use the following script. Note that this will take a while. The resulting plot (.png) and the raw data (.csv) are exported to `output/`.
``` bash
$ sh run_experiments.sh
```
To find the initial parameters for OpenCrowd with Gibbs Sampling, we used `src/non_feature_based/opencrowd_gibbs/experiment_3d.py`. This script generated the raw data for the 3d plots we show in the appendix.
