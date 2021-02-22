# Multiclass Classification OpenCrowd

### Installation
Scripts were successfully tested with Python 3.6.9.

#### Setup virtual environment
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