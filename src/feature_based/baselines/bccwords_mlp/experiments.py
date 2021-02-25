from feature_based.baselines.bccwords_mlp.bayesian_combination.ibcc import BCCWords
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.callbacks import EarlyStopping

def getBCCWordsModel(num_of_classes=3, num_of_workers=3, max_iter=3):
    return BCCWords(L=num_of_classes, K=num_of_workers, max_iter=max_iter, eps=1e-4,
             alpha0_diags=1.0, alpha0_factor=1.0, beta0_factor=1.0,
             verbose=False, use_lowerbound=False)

def getAnnotationMatrix(file_path):
    df = pd.read_csv(file_path,header=None)
    df['label'] = pd.factorize(df[2],sort=True)[0]
    l = []
    n_workers = df[0].unique().shape[0]
    for i in range(df[1].unique().shape[0]):
        t = [-1] * n_workers
        for index, row in df[df[1] == i].iterrows():
            t[row[0]] = row['label']
        l.append(t)
    return np.array(l)

def getBCCWordsFeatures(number_of_items):
    words = ['w1'] * number_of_items
    return np.array(words)

def getGroundTruth(file_path):
    df = pd.read_csv(file_path)
    ground_truth = pd.factorize(df['label'], sort=True)[0]
    return ground_truth

def getGoldLabels(supervision_rate, ground_truth):
    to_delete = ground_truth.shape[0] - int(ground_truth.shape[0] * supervision_rate)
    gold_labels = ground_truth.copy()
    gold_labels[-to_delete:] = -1
    return gold_labels

def getClassifierFeatures(file_path):
    classifier_features = pd.read_csv(file_path, header=None).values
    return classifier_features

def exp_iter(epochs, supervision_rate, ground_truth, aij, classifier_features, bccwords_features, file_out, iters=[]):
    true_labels_pr = pd.get_dummies(ground_truth).values
    x_train, x_test, y_train, y_test = train_test_split(classifier_features, true_labels_pr,
                                                        test_size=(round(1 - supervision_rate, 1)), shuffle=False)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)
    class_num = y_train.shape[1]
    input_dim = x_train.shape[1]
    gold_labels = getGoldLabels(supervision_rate,ground_truth)
    l = []
    for i in iters:
        test_accuracy = []
        val_accuracy = []
        test_auc = []
        val_auc = []
        for e in range(epochs):
            model = getBCCWordsModel(y_train.shape[1],aij.shape[1],i)
            probs, z_i, un = model.fit_predict(C=aij, features=bccwords_features, gold_labels=gold_labels, input_dim=input_dim,
                                           class_num=class_num, x_train=x_train, y_train=y_train, x_val=x_val,
                                           y_val=y_val, classifier_features=classifier_features)
            test_accuracy.append(accuracy_score(ground_truth[-x_test.shape[0]:], z_i[-x_test.shape[0]:]))
            val_accuracy.append(accuracy_score(ground_truth[x_train.shape[0]:-x_test.shape[0]], z_i[x_train.shape[0]:-x_test.shape[0]]))
            test_auc.append(roc_auc_score(true_labels_pr[-x_test.shape[0]:, :], probs[-x_test.shape[0]:, :], multi_class="ovo",
                                     average="macro"))
            val_auc.append(roc_auc_score(true_labels_pr[x_train.shape[0]:-x_test.shape[0], :],
                                    probs[x_train.shape[0]:-x_test.shape[0], :], multi_class="ovo",
                                    average="macro"))
        dct = {}
        dct['epochs'] = epochs
        dct['iter'] = i
        dct['test_accuracy'] = np.mean(test_accuracy)
        dct['test_auc'] = np.mean(test_auc)
        dct['val_accuracy'] = np.mean(val_accuracy)
        dct['val_auc'] = np.mean(val_auc)
        l.append(dct)
    pd.DataFrame(l).to_csv(file_out)

def exp_supervision(epochs, iter, ground_truth, aij, classifier_features, bccwords_features, file_out, supervision=[]):
    l = []
    true_labels_pr = pd.get_dummies(ground_truth).values
    for supervision_rate in supervision:
        test_accuracy = []
        val_accuracy = []
        test_auc = []
        val_auc = []
        gold_labels = getGoldLabels(supervision_rate, ground_truth)
        x_train, x_test, y_train, y_test = train_test_split(classifier_features, true_labels_pr,
                                                        test_size=(round(1 - supervision_rate, 1)), shuffle=False)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)
        class_num = y_train.shape[1]
        input_dim = x_train.shape[1]
        for e in range(epochs):
            model = getBCCWordsModel(y_train.shape[1], aij.shape[1], iter)
            probs, z_i, un = model.fit_predict(C=aij, features=bccwords_features, gold_labels=gold_labels, input_dim=input_dim,
                                           class_num=class_num, x_train=x_train, y_train=y_train, x_val=x_val,
                                           y_val=y_val, classifier_features=classifier_features)


            test_accuracy.append(accuracy_score(ground_truth[-x_test.shape[0]:], z_i[-x_test.shape[0]:]))
            val_accuracy.append(accuracy_score(ground_truth[x_train.shape[0]:-x_test.shape[0]], z_i[x_train.shape[0]:-x_test.shape[0]]))

            test_auc.append(roc_auc_score(true_labels_pr[-x_test.shape[0]:, :], probs[-x_test.shape[0]:, :], multi_class="ovo",
                                     average="macro"))
            val_auc.append(roc_auc_score(true_labels_pr[x_train.shape[0]:-x_test.shape[0], :],
                                    probs[x_train.shape[0]:-x_test.shape[0], :], multi_class="ovo",
                                    average="macro"))
        dct = {}
        dct['epochs'] = epochs
        dct['iter'] = iter
        dct['supervision_rate'] = supervision_rate
        dct['test_accuracy'] = np.mean(test_accuracy)
        dct['test_auc'] = np.mean(test_auc)
        dct['val_accuracy'] = np.mean(val_accuracy)
        dct['val_auc'] = np.mean(val_auc)
        l.append(dct)
    result = pd.DataFrame(l)
    result.to_csv(file_out)
    return result

def exp_supervision_mlp(epochs, ground_truth, classifier_features, file_out, supervision=[]):
    l = []
    true_labels_pr = pd.get_dummies(ground_truth).values
    model = BCCWords()
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10,
                            verbose=0, mode='auto', restore_best_weights=True)
    for supervision_rate in supervision:
        test_accuracy = []
        val_accuracy = []
        test_auc = []
        val_auc = []
        x_train, x_test, y_train, y_test = train_test_split(classifier_features, true_labels_pr,
                                                        test_size=(round(1 - supervision_rate, 1)), shuffle=False)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)
        class_num = y_train.shape[1]
        input_dim = x_train.shape[1]
        for e in range(epochs):
            classifier = model.define_multiclass_nn(input_dim,class_num)
            classifier.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=[monitor], verbose=0, epochs=100,
                           batch_size=4)
            theta_i = classifier.predict(classifier_features)
            theta_i_label = np.argmax(theta_i, axis=1)
            test_accuracy.append(accuracy_score(ground_truth[-x_test.shape[0]:], theta_i_label[-x_test.shape[0]:]))
            val_accuracy.append(accuracy_score(ground_truth[x_train.shape[0]:-x_test.shape[0]], theta_i_label[x_train.shape[0]:-x_test.shape[0]]))

            test_auc.append(roc_auc_score(true_labels_pr[-x_test.shape[0]:, :], theta_i[-x_test.shape[0]:, :], multi_class="ovo",
                                     average="macro"))
            val_auc.append(roc_auc_score(true_labels_pr[x_train.shape[0]:-x_test.shape[0], :],
                                    theta_i[x_train.shape[0]:-x_test.shape[0], :], multi_class="ovo",
                                    average="macro"))
        dct = {}
        dct['epochs'] = epochs
        dct['supervision_rate'] = supervision_rate
        dct['test_accuracy'] = np.mean(test_accuracy)
        dct['test_auc'] = np.mean(test_auc)
        dct['val_accuracy'] = np.mean(val_accuracy)
        dct['val_auc'] = np.mean(val_auc)
        l.append(dct)
    result = pd.DataFrame(l)
    result.to_csv(file_out)
    return result

def exp_supervision_lr(epochs, ground_truth, classifier_features, file_out, C, supervision=[]):
    l = []
    true_labels_pr = pd.get_dummies(ground_truth).values
    for supervision_rate in supervision:
        test_accuracy = []
        val_accuracy = []
        test_auc = []
        val_auc = []
        x_train, x_test, y_train, y_test = train_test_split(classifier_features, ground_truth,
                                                        test_size=(round(1 - supervision_rate, 1)), shuffle=False)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)
        for e in range(epochs):
            classifier = LogisticRegression(max_iter=1000,C=C).fit(x_train, y_train)
            theta_i = classifier.predict_proba(classifier_features)
            theta_i_label = classifier.predict(classifier_features)
            test_accuracy.append(accuracy_score(ground_truth[-x_test.shape[0]:], theta_i_label[-x_test.shape[0]:]))
            val_accuracy.append(accuracy_score(ground_truth[x_train.shape[0]:-x_test.shape[0]], theta_i_label[x_train.shape[0]:-x_test.shape[0]]))

            test_auc.append(roc_auc_score(true_labels_pr[-x_test.shape[0]:, :], theta_i[-x_test.shape[0]:, :], multi_class="ovo",
                                     average="macro"))
            val_auc.append(roc_auc_score(true_labels_pr[x_train.shape[0]:-x_test.shape[0], :],
                                    theta_i[x_train.shape[0]:-x_test.shape[0], :], multi_class="ovo",
                                    average="macro"))
        dct = {}
        dct['epochs'] = epochs
        dct['C'] = C
        dct['supervision_rate'] = supervision_rate
        dct['test_accuracy'] = np.mean(test_accuracy)
        dct['test_auc'] = np.mean(test_auc)
        dct['val_accuracy'] = np.mean(val_accuracy)
        dct['val_auc'] = np.mean(val_auc)
        l.append(dct)
    result = pd.DataFrame(l)
    result.to_csv(file_out)
    return result
