import GLAD.method as glad
import CATD.method as catd
import LFC.method as lfc
import DS.method as ds
import MV.method as mv
import utils
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

epochs = 10
supervision_rate = 0.6
sampling_rate = 0

def evaluate_supervision_rate(method,supervision_rate,filepath):
	l = []
	for rate in supervision_rate:
		train_size = int(true_labels.shape[0] * rate)
		test_size = int((true_labels.shape[0] * (1-rate))/2)
		e2t = utils.gete2t(train_size,true_labels)
		all_labels = np.unique(true_labels)

		all_accuracy_test = []
		all_auc_test = []
		all_accuracy_val = []
		all_auc_val = []
		print('rate', rate)
		for i in range(epochs):
			e2wl,w2el,label_set = utils.gete2wlandw2el(annotation_file,sampling_rate)
			e2lpd = method.run(e2t,e2wl,w2el,label_set)

			df = pd.DataFrame(utils.getLabels(e2lpd,method.__name__))
			df = df.sort_values(by=['example'])
			result_labels = df['label'].values

			if method.__name__ == 'CATD.method':
				result_labels_encoded = pd.get_dummies(result_labels).values
			else:
				probs = utils.getLabelProbabilities(e2lpd,method.__name__,all_labels)
				probs = pd.DataFrame(probs).sort_values(by=['example'])
				result_labels_encoded = pd.DataFrame(probs.label_prob.tolist(), index= probs.index).values

			accuracy_test = accuracy_score(true_labels[-test_size:], result_labels[-test_size:])
			accuracy_val = accuracy_score(true_labels[train_size:-test_size], result_labels[train_size:-test_size])
			auc_test = roc_auc_score(true_labels_encoded[-test_size:], result_labels_encoded[-test_size:],multi_class="ovo",average="macro")
			auc_val = roc_auc_score(true_labels_encoded[train_size:-test_size], result_labels_encoded[train_size:-test_size],multi_class="ovo",average="macro")

			all_accuracy_test.append(accuracy_test)
			all_auc_test.append(auc_test)
			all_accuracy_val.append(accuracy_val)
			all_auc_val.append(auc_val)
			print('accuracy_test',accuracy_test)

		dct = {}
		dct['supervision_rate'] = rate
		dct['sampling_rate'] = sampling_rate
		dct['epochs'] = epochs
		dct['mean_accuracy_test'] = np.mean(all_accuracy_test)
		dct['mean_auc_test'] = np.mean(all_auc_test)
		dct['mean_accuracy_val'] = np.mean(all_accuracy_val)
		dct['mean_auc_val'] = np.mean(all_auc_val)
		l.append(dct)
	pd.DataFrame(l).to_csv(filepath,index=False)

def evaluate_sampling_rate(method,sampling_rate,filepath):
	train_size = int(true_labels.shape[0] * supervision_rate)
	test_size = int((true_labels.shape[0] * (1-supervision_rate))/2)
	e2t = utils.gete2t(train_size,true_labels)

	l = []
	for rate in sampling_rate:
		all_accuracy_test = []
		all_auc_test = []
		all_accuracy_val = []
		all_auc_val = []
		print('rate',rate)
		for i in range(epochs):
			e2wl,w2el,label_set = utils.gete2wlandw2el(annotation_file,rate)
			e2lpd = method.run(e2t,e2wl,w2el,label_set)

			df = pd.DataFrame(utils.getLabels(e2lpd,method.__name__))
			df = df.sort_values(by=['example'])
			result_labels = df['label'].values

			accuracy_test = accuracy_score(true_labels[-test_size:], result_labels[-test_size:])
			accuracy_val = accuracy_score(true_labels[train_size:-test_size], result_labels[train_size:-test_size])
			result_labels_encoded = pd.get_dummies(result_labels).values
			auc_test = roc_auc_score(true_labels_encoded[-test_size:], result_labels_encoded[-test_size:],multi_class="ovo",average="macro")
			auc_val = roc_auc_score(true_labels_encoded[train_size:-test_size], result_labels_encoded[train_size:-test_size],multi_class="ovo",average="macro")

			all_accuracy_test.append(accuracy_test)
			all_auc_test.append(auc_test)
			all_accuracy_val.append(accuracy_val)
			all_auc_val.append(auc_val)
			print('accuracy_test',accuracy_test)

		dct = {}
		dct['supervision_rate'] = supervision_rate
		dct['sampling_rate'] = rate
		dct['epochs'] = epochs
		dct['mean_accuracy_test'] = np.mean(all_accuracy_test)
		dct['mean_auc_test'] = np.mean(all_auc_test)
		dct['mean_accuracy_val'] = np.mean(all_accuracy_val)
		dct['mean_auc_val'] = np.mean(all_auc_val)
		l.append(dct)
	pd.DataFrame(l).to_csv(filepath)


truth_file = '../input/sentiment_transformed_labels.csv'
true_labels = pd.read_csv(truth_file)
true_labels = true_labels['label_code'].values
true_labels_encoded = pd.get_dummies(true_labels).values

### sentiment
annotation_file = '../input/sentiment_transformed_aij.csv'
# evaluate_supervision_rate(mv,[0.3,0.4,0.5,0.6,0.7,0.8],'output/mv_sentiment_supervision_rate.csv')
# evaluate_supervision_rate(ds,[0.3,0.4,0.5,0.6,0.7,0.8],'output/ds_sentiment_supervision_rate.csv')
# evaluate_supervision_rate(catd,[0.3,0.4,0.5,0.6,0.7,0.8],'output/catd_sentiment_supervision_rate.csv')
# evaluate_supervision_rate(lfc,[0.3,0.4,0.5,0.6,0.7,0.8],'output/lfc_sentiment_supervision_rate.csv')
# evaluate_supervision_rate(glad,[0.3,0.4,0.5,0.6,0.7,0.8],'output/glad_sentiment_supervision_rate.csv')

### sentiment sparse
annotation_file = '../input/sentiment_sparse_transformed_aij.csv'
# evaluate_supervision_rate(mv,[0.3,0.4,0.5,0.6,0.7,0.8],'output/mv_sentiment_sparse_supervision_rate.csv')
# evaluate_supervision_rate(ds,[0.3,0.4,0.5,0.6,0.7,0.8],'output/ds_sentiment_sparse_supervision_rate.csv')
# evaluate_supervision_rate(catd,[0.3,0.4,0.5,0.6,0.7,0.8],'output/catd_sentiment_sparse_supervision_rate.csv')
# evaluate_supervision_rate(lfc,[0.3,0.4,0.5,0.6,0.7,0.8],'output/lfc_sentiment_sparse_supervision_rate.csv')
# evaluate_supervision_rate(glad,[0.3,0.4,0.5,0.6,0.7,0.8],'output/glad_sentiment_sparse_supervision_rate.csv')

### influencer
truth_file = '../input/influencer_transformed_labels.csv'
true_labels = pd.read_csv(truth_file)
true_labels = true_labels['label_code'].values
true_labels_encoded = pd.get_dummies(true_labels).values
annotation_file = '../input/influencer_transformed_aij.csv'
# evaluate_supervision_rate(mv,[0.3,0.4,0.5,0.6,0.7,0.8],'output/mv_influencer_supervision_rate.csv')
# evaluate_supervision_rate(ds,[0.3,0.4,0.5,0.6,0.7,0.8],'output/ds_influencer_supervision_rate.csv')
# evaluate_supervision_rate(catd,[0.3,0.4,0.5,0.6,0.7,0.8],'output/catd_influencer_supervision_rate.csv')
# evaluate_supervision_rate(lfc,[0.3,0.4,0.5,0.6,0.7,0.8],'output/lfc_influencer_supervision_rate.csv')
# evaluate_supervision_rate(glad,[0.3,0.4,0.5,0.6,0.7,0.8],'output/glad_influencer_supervision_rate.csv')
