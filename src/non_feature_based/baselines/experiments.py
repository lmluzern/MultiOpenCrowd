from non_feature_based.baselines import utils
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

def evaluate_supervision_rate(epochs,method,supervision_rate,filepath,truth_file,annotation_file):
	sampling_rate = 0
	true_labels = pd.read_csv(truth_file)
	true_labels = true_labels['label_code'].values
	true_labels_encoded = pd.get_dummies(true_labels).values
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
		for i in range(epochs):
			e2wl,w2el,label_set = utils.gete2wlandw2el(annotation_file,sampling_rate)
			e2lpd = method.run(e2t,e2wl,w2el,label_set)

			df = pd.DataFrame(utils.getLabels(e2lpd,method.__name__))
			df = df.sort_values(by=['example'])
			result_labels = df['label'].values

			if 'CATD.method' in method.__name__:
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

		dct = {}
		dct['supervision_rate'] = rate
		dct['sampling_rate'] = sampling_rate
		dct['epochs'] = epochs
		dct['mean_accuracy_test'] = np.mean(all_accuracy_test)
		dct['mean_auc_test'] = np.mean(all_auc_test)
		dct['mean_accuracy_val'] = np.mean(all_accuracy_val)
		dct['mean_auc_val'] = np.mean(all_auc_val)
		l.append(dct)
	result = pd.DataFrame(l)
	result.to_csv(filepath,index=False)
	return result
