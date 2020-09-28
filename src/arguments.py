# default arguments
args = {
    'influencer_file_labeled' : '../input/multiclass_labeled.csv', 
    'annotation_file' : '../input/multiclass_aij.csv',
    'labels_file' : '../input/multiclass_labels.csv',
    'tweet2vec_file' : '../input/tweet2vec_glove_200d.csv',
    'tweet2vec_dim' : 200,
    'theta_file' : '../output/simple_example/theta/theta_i0_sup_60_sr_10_0_name.csv',
    'evaluation_file' : '../output/evaluation.csv',
    'weights_before_em' : '../output/simple_example/weights_before/weights_before_em_sup_60_sr_10_0_name.csv',
    'weights_after_em' : '../output/simple_example/weights_after/weights_after_em_sup_60_sr_10_0_name.csv',
    'total_epochs' : 100,
    'n_neurons' : 10,
    'steps' : 1,
    'supervision_rate' : 0.6,
    'iterr' : 20,
    'sampling_rate' : 1.0,
    'worker_reliability_file' : '../output/simple_example/worker_reliability/worker_reliability_sup_60_sr_10_0_name.csv',
    'influencer_quality_file' : '../output/simple_example/influencer_quality/influencer_quality_sup_60_sr_10_name.csv',
    'random_sampling' : True,
    'new_alpha_value' : 0.6
}
