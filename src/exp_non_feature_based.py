import sys
from non_feature_based.opencrowd_gibbs import experiments as gibbs_exp
from non_feature_based.baselines import experiments as baseline_exp
from non_feature_based.baselines.MV import method as mv
from non_feature_based.baselines.DS import method as ds
from non_feature_based.baselines.CATD import method as catd
from non_feature_based.baselines.LFC import method as lfc
from non_feature_based.baselines.GLAD import method as glad
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

if dataset == 'influencer':
    param = {
    'annotation_file' : '../input/influencer_aij.csv',
    'labels_file' : '../input/influencer_labels.csv',
    'A_0' : 1,
    'B_0' : 1,
    'gamma_0' : 0.8,
    'mu_0' : 3.5,
    'iters' : 1000,
    'burn_in_rate' : 0.2,
    'supervision_rate' : 0.6,
    'sampling_rate' : 0.0
    }
    truth_file = '../input/influencer_transformed_labels.csv'
    annotation_file = '../input/influencer_transformed_aij.csv'
elif dataset == 'sentiment_sparse':
    param = {
    'annotation_file' : '../input/sentiment_sparse_aij.csv',
    'labels_file' : '../input/sentiment_labels.csv',
    'A_0' : 8,
    'B_0' : 1,
    'gamma_0' : 0.1,
    'mu_0' : 1.0,
    'iters' : 1000,
    'burn_in_rate' : 0.2,
    'supervision_rate' : 0.6,
    'sampling_rate' : 0.0
    }
    truth_file = '../input/sentiment_transformed_labels.csv'
    annotation_file = '../input/sentiment_sparse_transformed_aij.csv'
else:
    param = {
    'annotation_file' : '../input/sentiment_aij.csv',
    'labels_file' : '../input/sentiment_labels.csv',
    'A_0' : 8,
    'B_0' : 1,
    'gamma_0' : 0.2,
    'mu_0' : 4.5,
    'iters' : 1000,
    'burn_in_rate' : 0.2,
    'supervision_rate' : 0.6,
    'sampling_rate' : 0.0
    }
    truth_file = '../input/sentiment_transformed_labels.csv'
    annotation_file = '../input/sentiment_transformed_aij.csv'

supervision_rate = [0.3,0.4,0.5,0.6,0.7,0.8]
epochs = 10
print('run OpenCrowd with Gibbs Sampling...')
gibbs_result = gibbs_exp.run_experiment(epochs=epochs,file_out='../output/gibbs_' + dataset + '_supervision_rate.csv',value_range=supervision_rate,value_name='supervision_rate',param=param)

print('run MV...')
mv_result = baseline_exp.evaluate_supervision_rate(epochs,mv,supervision_rate,'../output/mv_' + dataset +'_supervision_rate.csv',truth_file,annotation_file)
print('run D&S...')
ds_result = baseline_exp.evaluate_supervision_rate(epochs,ds,supervision_rate,'../output/ds_' + dataset + '_supervision_rate.csv',truth_file,annotation_file)
print('run CATD...')
catd_result = baseline_exp.evaluate_supervision_rate(epochs,catd,supervision_rate,'../output/catd_' + dataset + '_supervision_rate.csv',truth_file,annotation_file)
print('run LFC...')
lfc_result = baseline_exp.evaluate_supervision_rate(epochs,lfc,supervision_rate,'../output/lfc_' + dataset + '_supervision_rate.csv',truth_file,annotation_file)
print('run GLAD...')
glad_result = baseline_exp.evaluate_supervision_rate(epochs,glad,supervision_rate,'../output/glad_' + dataset + '_supervision_rate.csv',truth_file,annotation_file)

plt.xlabel("supervision rate")
plt.ylabel("accuracy")
plt.plot(lfc_result['supervision_rate'],lfc_result['mean_accuracy_test'],marker='o',label='LFC')
plt.plot(ds_result['supervision_rate'],ds_result['mean_accuracy_test'],marker='o',label='D&S')
plt.plot(glad_result['supervision_rate'],glad_result['mean_accuracy_test'],marker='o',label='GLAD')
plt.plot(catd_result['supervision_rate'],catd_result['mean_accuracy_test'],marker='o',label='CATD')
plt.plot(mv_result['supervision_rate'],mv_result['mean_accuracy_test'],marker='o',label='MV')
plt.plot(gibbs_result['supervision_rate'],gibbs_result['mean_accuracy_test'],marker='o',label='OpenCrowd w. Gibbs')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('../output/exp_non_feature_based_' + dataset + '_supervision_accuracy.png')

plt.clf()
plt.xlabel("supervision rate")
plt.ylabel("auc")
plt.plot(lfc_result['supervision_rate'],lfc_result['mean_auc_test'],marker='o',label='LFC')
plt.plot(ds_result['supervision_rate'],ds_result['mean_auc_test'],marker='o',label='D&S')
plt.plot(glad_result['supervision_rate'],glad_result['mean_auc_test'],marker='o',label='GLAD')
plt.plot(catd_result['supervision_rate'],catd_result['mean_auc_test'],marker='o',label='CATD')
plt.plot(mv_result['supervision_rate'],mv_result['mean_auc_test'],marker='o',label='MV')
plt.plot(gibbs_result['supervision_rate'],gibbs_result['mean_auc_test'],marker='o',label='OpenCrowd w. Gibbs')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.savefig('../output/exp_non_feature_based_' + dataset + '_supervision_auc.png')
