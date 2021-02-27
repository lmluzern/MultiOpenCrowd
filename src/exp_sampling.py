from feature_based.multiclass_opencrowd import experiments as multiclass_opencrowd_exp
import matplotlib.pyplot as plt

datasets = ['influencer','sentiment_sparse','sentiment']
supervision_rate = 0.6
epochs = 10
for random_sampling in [True,False]:
    plt.figure(1)
    plt.clf()
    plt.xlabel("sampling rate")
    plt.ylabel("accuracy")
    plt.figure(2)
    plt.xlabel("sampling rate")
    plt.ylabel("auc")
    plt.clf()
    if random_sampling:
        sampling_type = 'random_sampling'
        sampling_rate = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    else:
        sampling_type = 'negative_sampling'
        sampling_rate = [0,1,2,3,4,5,6,7,8,9,10]

    for dataset in datasets:
        if dataset == 'influencer':
            filename = dataset
            iter_opencrowd = 5
        else:
            filename = 'sentiment'
            iter_opencrowd = 6

        print('run',sampling_type,'on',dataset,'...')
        multiclass_opencrowd_result = multiclass_opencrowd_exp.run_experiment(epochs=epochs,
                                                                              file_out='../output/multiclass_opencrowd_' + dataset + '_' + sampling_type + '.csv',
                                                                              value_range=sampling_rate,
                                                                              value_name='sampling_rate',
                                                                              random_sampling=random_sampling,
                                                                              sampling_rate=0,
                                                                              supervision_rate=0.6, iterr=iter_opencrowd,
                                                                              file_labeled='../input/' + filename + '_labeled.csv',
                                                                              annotation_file='../input/' + dataset + '_aij.csv',
                                                                              labels_file='../input/' + filename + '_labels.csv',
                                                                              tweet2vec_file='../input/' + filename + '_tweet2vec_glove_200d.csv')
        plt.figure(1)
        plt.plot(multiclass_opencrowd_result['sampling_rate'], multiclass_opencrowd_result['accuracy_theta_i_test'],
                 marker='o', label=dataset)
        plt.figure(2)
        plt.plot(multiclass_opencrowd_result['sampling_rate'], multiclass_opencrowd_result['auc_theta_i_test'],
                 marker='o', label=dataset)

    plt.figure(1)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig('../output/exp_feature_based_' + sampling_type + '_accuracy.png')
    plt.figure(2)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig('../output/exp_feature_based_' + sampling_type + '_auc.png')
