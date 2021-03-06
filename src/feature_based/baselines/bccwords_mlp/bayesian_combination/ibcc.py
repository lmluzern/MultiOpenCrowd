'''
Wrapper classes to implement independent Bayesian classifier combination (IBCC-VB)
using the Bayesian combination framework, along with variants of that model.
'''
from feature_based.baselines.bccwords_mlp.bayesian_combination.bayesian_combination import BC


class BCCWords(BC):
    def __init__(self, L=3, K=5, max_iter=20, eps=1e-4,
                     alpha0_diags=1.0, alpha0_factor=1.0, beta0_factor=1.0,
                     verbose=False, use_lowerbound=True):

        super().__init__(L, K, max_iter, eps,
                         alpha0_diags=alpha0_diags, alpha0_factor=alpha0_factor, beta0_factor=beta0_factor,
                         annotator_model='ibcc', taggers=None, true_label_model='noHMM', discrete_feature_likelihoods=True,
                         verbose=verbose, use_lowerbound=use_lowerbound)