python3 ../src/var_em.py \
		--labeled_social_features '../input/multiclass_labeled.csv' \
		--unlabeled_social_features 'NA'\
		--annotation_matrix '../input/multiclass_aij.csv'\
		--labels '../input/multiclass_labels.csv'\
		--total_epochs_nn 100\
		--total_neurons_nn 10\
		--steps 1\
		--supervision_rate 0.3\
		--nb_iterations_EM 20\
		--sampling_rate 10.0\
		--worker_reliability_file '../output/simple_example/worker_reliability/worker_reliability_sup_60_sr_10_0_name.csv'\
		--influencer_quality_file '../output/simple_example/influencer_quality/influencer_quality_sup_60_sr_10_name.csv'\
		--evaluation_file '../output/simple_example/evaluation/evaluation_sup_60_sr_10_0_name.csv'\
		--theta_file '../output/simple_example/theta/theta_i0_sup_60_sr_10_0_name.csv'\
		--weights_before_em '../output/simple_example/weights_before/weights_before_em_sup_60_sr_10_0_name.csv'\
		--weights_after_em '../output/simple_example/weights_after/weights_after_em_sup_60_sr_10_0_name.csv'\
		--tweet2vec '../input/tweet2vec_glove_200d.csv'\
		--tweet2vec_dim 200\