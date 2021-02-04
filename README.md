# Multiclass Classification of Open-ended Answers using OpenCrowd

### Installation
This package requires python3. To install python3, please check the official python website
https://www.python.org/downloads/

#### Python Libraries

``` bash 
$ pip3 install pandas keras sklearn tensorflow
```

## Usage
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