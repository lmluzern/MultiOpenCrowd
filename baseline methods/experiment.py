import GLAD.method as glad
import utils
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

supervision_rate = 0.3
sampling_rate = 2
annotation_file = '../input/transformed_multiclass_aij.csv'
truth_file = '../input/transformed_multiclass_labels.csv'

true_labels = pd.read_csv(truth_file)
true_labels = true_labels['label_code'].values

train_size = int(true_labels.shape[0] * supervision_rate)
test_size = int((true_labels.shape[0] * (1-supervision_rate))/2)

print('train_size',train_size)
print('test_size', test_size)

e2t = utils.gete2t(train_size,true_labels)
e2wl,w2el,label_set = utils.gete2wlandw2el(annotation_file,sampling_rate)

e2lpd = glad.run(e2t,e2wl,w2el,label_set)

df = pd.DataFrame(utils.getLabels(e2lpd))
df = df.sort_values(by=['example'])
result_labels = df['label'].values

print('auccracy (complete dataset):', accuracy_score(true_labels, result_labels))
print('auccracy (test dataset):', accuracy_score(true_labels[-test_size:], result_labels[-test_size:]))
print('auccracy (val dataset):', accuracy_score(true_labels[train_size:-test_size], result_labels[train_size:-test_size]))

true_labels_encoded = pd.get_dummies(true_labels).values
result_labels_encoded = pd.get_dummies(result_labels).values
roc = roc_auc_score(true_labels_encoded[-test_size:], result_labels_encoded[-test_size:],multi_class="ovo",average="macro")

print('roc',roc)

# orginal solution
truthfile = r'../input/transformed_multiclass_labels.csv'
accuracy = glad.getaccuracy(truthfile, e2lpd, label_set)
print('accuracy (complete dataset)',accuracy)