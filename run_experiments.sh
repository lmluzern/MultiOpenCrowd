#!/bin/bash
cd src/
python exp_non_feature_based.py influencer
python exp_non_feature_based.py sentiment_sparse
python exp_non_feature_based.py sentiment
python exp_feature_based.py influencer
python exp_feature_based.py sentiment_sparse
python exp_feature_based.py sentiment