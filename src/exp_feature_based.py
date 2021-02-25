import sys
from feature_based.baselines.bccwords_mlp import experiments as bccwords_mlp_exp
from feature_based.multiclass_opencrowd import experiments as multiclass_opencrowd_exp

import matplotlib.pyplot as plt

datasets = ['influencer','sentiment_sparse','sentiment']
try:
    dataset = sys.argv[1]
    if dataset not in datasets:
        print('invalid dataset - use: ' + str(datasets))
        exit()
except:
    print('dataset command line argument missing.')
    exit()

filename = dataset
C = 0.2 #LR parameter; evaluated [0.00001, 0.0001, ..., 10] on validation set
if dataset == 'influencer':
    sampling_rate = 6
    iter_opencrowd = 5
    iter = 5
    C = 0.06

elif dataset == 'sentiment_sparse':
    sampling_rate = 4
    iter_opencrowd = 20
    iter = 1
    filename = 'sentiment'

else:
    sampling_rate = 0.2
    iter_opencrowd = 10
    iter = 3

supervision_rate = [0.3,0.4,0.5,0.6,0.7,0.8]
epochs = 10
ground_truth = bccwords_mlp_exp.getGroundTruth('../input/' + filename + '_labels.csv')
aij = bccwords_mlp_exp.getAnnotationMatrix('../input/' + dataset + '_aij.csv')
classifier_features = bccwords_mlp_exp.getClassifierFeatures('../input/' + filename + '_features.csv')
bccwords_features = bccwords_mlp_exp.getBCCWordsFeatures(aij.shape[0])

print('run Multiclass OpenCrowd...')
multiclass_opencrowd_result = multiclass_opencrowd_exp.run_experiment(epochs=epochs, file_out='../output/multiclass_opencrowd_' + dataset + '_supervision_rate.csv', value_range=supervision_rate, value_name='supervision_rate', random_sampling=False, sampling_rate=sampling_rate, supervision_rate=0, iterr=iter_opencrowd, file_labeled='../input/' + filename + '_labeled.csv', annotation_file='../input/' + dataset + '_aij.csv', labels_file='../input/' + filename + '_labels.csv', tweet2vec_file='../input/' + filename + '_tweet2vec_glove_200d.csv')
print('run BCCWords MLP...')
bccwords_mlp_result = bccwords_mlp_exp.exp_supervision(epochs,iter,ground_truth,aij,classifier_features,bccwords_features,'../output/bccwords_mlp_' + dataset + '_supervision_rate.csv',supervision_rate)
print('run MLP...')
mlp_result = bccwords_mlp_exp.exp_supervision_mlp(epochs, ground_truth, classifier_features, '../output/mlp_' + dataset + '_supervision_rate.csv',supervision_rate)
print('run LR...')
lr_result = bccwords_mlp_exp.exp_supervision_lr(1, ground_truth, classifier_features, '../output/lr_' + dataset + '_supervision_rate.csv',C,supervision_rate)


plt.xlabel("supervision rate")
plt.ylabel("accuracy")
plt.plot(multiclass_opencrowd_result['supervision_rate'],multiclass_opencrowd_result['accuracy_theta_i_test'],marker='o',label='Mutli. OpenCrowd')
plt.plot(bccwords_mlp_result['supervision_rate'],bccwords_mlp_result['test_accuracy'],marker='o',label='BCCWords MLP')
plt.plot(mlp_result['supervision_rate'],mlp_result['test_accuracy'],marker='o',label='MLP')
plt.plot(lr_result['supervision_rate'],lr_result['test_accuracy'],marker='o',label='LR')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('../output/exp_feature_based_' + dataset + '_supervision_accuracy.png')

plt.clf()
plt.xlabel("supervision rate")
plt.ylabel("auc")
plt.plot(multiclass_opencrowd_result['supervision_rate'],multiclass_opencrowd_result['auc_theta_i_test'],marker='o',label='Mutli. OpenCrowd')
plt.plot(bccwords_mlp_result['supervision_rate'],bccwords_mlp_result['test_auc'],marker='o',label='BCCWords MLP')
plt.plot(mlp_result['supervision_rate'],mlp_result['test_auc'],marker='o',label='MLP')
plt.plot(lr_result['supervision_rate'],lr_result['test_auc'],marker='o',label='LR')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.savefig('../output/exp_feature_based_' + dataset + '_supervision_auc.png')
