import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.special import digamma
from numpy import linalg as LA
from nn_em import nn_em
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import random
import arguments

LABEL_NAMES = ['emerging', 'established', 'no_option']
NUMBER_OF_LABELS = len(LABEL_NAMES)
LABEL_INDEX = np.array(range(0,NUMBER_OF_LABELS))


def init_probabilities(n_infls):
    # initialize probability z_i (item's quality) randomly
    qz = (1.0/NUMBER_OF_LABELS) * np.ones((n_infls, NUMBER_OF_LABELS))
    # initialize probability alpha beta (worker's reliability)
    A = 2
    B = 2
    return qz, A, B


def init_alpha_beta(A, B, n_workers):
    alpha = np.zeros((n_workers, 1),dtype='float32')
    beta = np.zeros((n_workers, 1),dtype='float32')
    for w in range(0, n_workers):
        alpha[w] = A
        beta[w] = B
    return alpha, beta


def e_step(y_train, n_workers, q_z_i, annotation_matrix, alpha, beta, theta_i,true_labels,new_order,y_val,start_val,end_val,new_alpha_value,max_it=20):
    old_q_z_i = theta_i.copy()
    old_alpha = alpha.copy()
    old_beta = beta.copy()
    diff = []
    train_acc = []
    y_val_label = np.argmax(y_val,axis=1)

    for it in range(max_it):
        # update q(z)

        for infl in new_order.tolist():
            index_infl = np.where(new_order == infl)[0][0]
            assert infl == index_infl
            updated_q_z_i = theta_i[index_infl].copy()
            infl_aij = annotation_matrix[annotation_matrix[:, 1] == infl].copy()
            worker_answers = infl_aij[~np.all(infl_aij[:,2:] == 0, axis=1)]
            worker_n_answers = infl_aij[np.all(infl_aij[:,2:] == 0, axis=1)]
            T_i = worker_answers[:, 0]
            for worker in T_i.astype(int):
                w_answer = worker_answers[worker_answers[:, 0] == worker][:, 2:]
                w_answer_i = np.where(w_answer[0] == 1)[0][0]
                alpha_val = alpha[worker]
                beta_val =  beta[worker]
                updated_q_z_i[w_answer_i] = updated_q_z_i[w_answer_i] * np.exp(digamma(alpha_val) - digamma(alpha_val + beta_val))

                for no_answer_i in np.delete(LABEL_INDEX,w_answer_i):
                    updated_q_z_i[no_answer_i] = updated_q_z_i[no_answer_i] * np.exp(digamma(beta_val) - digamma(alpha_val + beta_val))

            T_i_n = worker_n_answers[:, 0]
            for worker in T_i_n.astype(int):
                alpha_val = alpha[worker]
                beta_val =  beta[worker]
                for no_answer_i in LABEL_INDEX:
                    updated_q_z_i[no_answer_i] = updated_q_z_i[no_answer_i] * np.exp(digamma(beta_val) - digamma(alpha_val + beta_val))

            # normalize
            new_q_z_i = updated_q_z_i * 1.0 / (updated_q_z_i.sum())
            q_z_i[index_infl] = new_q_z_i.copy()

        q_z_i = np.concatenate((y_train, q_z_i[y_train.shape[0]:]))

        # update q(r)
        new_alpha = np.zeros((n_workers, 1))
        new_beta = np.zeros((n_workers, 1))
        for worker in range(0, n_workers):
            new_alpha[worker] = alpha[worker]
            new_beta[worker] = beta[worker]

        for worker in range(0, n_workers):
            worker_aij = annotation_matrix[annotation_matrix[:, 0] == worker].copy()
            T_j = worker_aij[~np.all(worker_aij[:,2:] == 0, axis=1)]
            T_j_n = worker_aij[np.all(worker_aij[:,2:] == 0, axis=1)]
            for infl in T_j[:, 1].astype(int):
                index_infl = np.where(new_order == infl)[0][0]
                assert infl == index_infl
                worker_answer = T_j[T_j[:, 1] == infl][:, 2:]
                worker_answer_i = np.where(worker_answer[0] == 1)[0][0]
                new_alpha[worker] += q_z_i[index_infl][worker_answer_i]
                new_beta[worker] += 1 - q_z_i[index_infl][worker_answer_i]

            for infl in T_j_n[:, 1].astype(int):
                new_alpha[worker] += new_alpha_value
                new_beta[worker] += 1 - new_alpha_value


        for worker in range(0, n_workers):
            alpha[worker] = new_alpha[worker]
            beta[worker] = new_beta[worker]

        q_z_i_change = LA.norm(old_q_z_i - q_z_i)
        # da = LA.norm(old_alpha - alpha)
        # db = LA.norm(old_beta - beta)

        old_q_z_i = q_z_i.copy()
        old_alpha = alpha.copy()
        old_beta = beta.copy()

        q_z_i_val_label = np.argmax(q_z_i[start_val:end_val],axis=1)
        q_z_i_acc = accuracy_score(y_val_label,q_z_i_val_label)

        diff.append(q_z_i_change)
        train_acc.append(q_z_i_acc)

        print(it, q_z_i_change)

        if q_z_i_change < 0.1:
            break

    return q_z_i,alpha,beta

def m_step(nn_em,q_z_i, classifier, social_features, total_epochs, steps, y_test, y_val,X_val,start_val,alpha, beta):
    theta_i, classifier, weights = nn_em.train_m_step_early_stopp(classifier, social_features,
                                                      q_z_i,
                                                      steps, total_epochs, y_test, y_val,X_val,start_val)
    return theta_i,classifier


def var_em(nn_em_in, n_infls_label,aij_s,new_order, n_workers, social_features_labeled, true_labels, supervision_rate, \
           column_names, n_neurons, m_feats, weights_before_em,weights_after_em,iterr,total_epochs,evaluation_file,theta_file,steps,new_alpha_value,multiple_input,tweet2vec_dim):
    n_infls = n_infls_label
    q_z_i, A, B = init_probabilities(n_infls)

    alpha, beta = init_alpha_beta(A, B, n_workers)

    X_train, X_test, y_train, y_test = train_test_split(social_features_labeled, true_labels,
                                                        test_size=(1 - supervision_rate), shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)

    social_features = social_features_labeled

    start_val = X_train.shape[0]
    end_val = X_train.shape[0] + X_val.shape[0]

    n_stat_feats = m_feats - tweet2vec_dim

    if multiple_input:
        n_neurons = int((NUMBER_OF_LABELS + n_stat_feats)/2)
        classifier = nn_em_in.create_multiple_input_model_mlp(n_neurons,(n_stat_feats,),(tweet2vec_dim,),NUMBER_OF_LABELS)
        # classifier = nn_em_in.create_multiple_input_model(n_neurons,(n_stat_feats,),(tweet2vec_dim,1),NUMBER_OF_LABELS)

        X_train = [X_train[:,:n_stat_feats],X_train[:,n_stat_feats:]]
        X_val = [X_val[:,:n_stat_feats],X_val[:,n_stat_feats:]]
        X_test = [X_test[:,:n_stat_feats],X_test[:,n_stat_feats:]]

        social_features = [social_features[:,:n_stat_feats],social_features[:,n_stat_feats:]]
        # X_train = [X_train[:,:n_stat_feats],X_train[:,n_stat_feats:].reshape(X_train[:,n_stat_feats:].shape[0], X_train[:,n_stat_feats:].shape[1], 1)]
        # X_val = [X_val[:,:n_stat_feats],X_val[:,n_stat_feats:].reshape(X_val[:,n_stat_feats:].shape[0], X_val[:,n_stat_feats:].shape[1], 1)]
        # X_test = [X_test[:,:n_stat_feats],X_test[:,n_stat_feats:].reshape(X_test[:,n_stat_feats:].shape[0], X_test[:,n_stat_feats:].shape[1], 1)]

        # social_features = [social_features[:,:n_stat_feats],social_features[:,n_stat_feats:].reshape(social_features[:,n_stat_feats:].shape[0], social_features[:,n_stat_feats:].shape[1], 1)]
    else:
        n_neurons = int((NUMBER_OF_LABELS + m_feats)/2)
        classifier = nn_em_in.define_multiclass_nn(n_neurons,m_feats,NUMBER_OF_LABELS)
    print(classifier.summary())

    steps_it0 = 0
    epsilon = 1e-4
    theta_i = q_z_i.copy()
    old_theta_i = np.zeros((n_infls, NUMBER_OF_LABELS))

    y_val_label = np.argmax(y_val,axis=1)
    y_test_label = np.argmax(y_test,axis=1)
    y_train_label = np.argmax(y_train,axis=1)

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, 
                        verbose=0, mode='auto', restore_best_weights=True)

    classifier.fit(X_train, y_train, validation_data=(X_val,y_val), callbacks=[monitor], verbose=2, epochs=100, batch_size=4)

    theta_i_val = classifier.predict(X_val)
    theta_i_test = classifier.predict(X_test)

    theta_i_val_label = np.argmax(theta_i_val,axis=1)
    theta_i_test_label = np.argmax(theta_i_test,axis=1)
    
    weights = classifier.get_weights()
    # pd.DataFrame(np.concatenate((column_names[1:], weights[0]), axis=1)).to_csv(weights_before_em, encoding="utf-8")

    auc_val = roc_auc_score(y_val, theta_i_val,multi_class="ovo",average="macro")
    auc_test = roc_auc_score(y_test, theta_i_test,multi_class="ovo",average="macro")

    print('Classification Repport for validation set:\n', classification_report(y_val_label, theta_i_val_label, target_names=LABEL_NAMES))
    print('auc_val:', auc_val)
    print('Classification Repport for test set:\n', classification_report(y_test_label, theta_i_test_label, target_names=LABEL_NAMES))
    print('auc_test:', auc_test)

    theta_i = np.concatenate((y_train, theta_i_val, theta_i_test))
    theta_quality = np.concatenate((true_labels, theta_i), axis=1)
    pd.DataFrame(theta_quality).to_csv(theta_file, index=False)
    
    accuracy_theta_i_test = [] 
    accuracy_theta_i_val = []
    accuracy_q_z_i_test = []
    accuracy_q_z_i_val = []

    auc_theta_i_test = []

    em_step = 0
    while em_step < iterr:
        # variational E step
        q_z_i, alpha, beta = e_step(y_train, n_workers, q_z_i, aij_s, alpha,
                                               beta, theta_i, true_labels,new_order,y_val,start_val,end_val,new_alpha_value)

        # variational M step
        theta_i, classifier = m_step(nn_em_in, q_z_i, classifier, social_features, total_epochs, steps, y_test, y_val, X_val,
                                     start_val, alpha, beta)
        em_step += 1

        q_z_i_val_label = np.argmax(q_z_i[start_val:end_val],axis=1)
        q_z_i_test_label = np.argmax(q_z_i[end_val:],axis=1)

        auc_val = roc_auc_score(y_val, q_z_i[start_val:end_val],multi_class="ovo",average="macro")
        auc_test = roc_auc_score(y_test, q_z_i[end_val:],multi_class="ovo",average="macro")

        theta_i_val_label = np.argmax(theta_i[start_val:end_val],axis=1)
        theta_i_test_label = np.argmax(theta_i[end_val:],axis=1)

        auc_val_theta = roc_auc_score(y_val, theta_i[start_val:end_val],multi_class="ovo",average="macro")
        auc_test_theta = roc_auc_score(y_test, theta_i[end_val:],multi_class="ovo",average="macro")

        accuracy_theta_i_test.append(accuracy_score(y_test_label, theta_i_test_label))
        accuracy_theta_i_val.append(accuracy_score(y_val_label, theta_i_val_label))

        accuracy_q_z_i_test.append(accuracy_score(y_test_label, q_z_i_test_label))
        accuracy_q_z_i_val.append(accuracy_score(y_val_label, q_z_i_val_label))

        auc_theta_i_test.append(auc_test_theta)

        print('em_step', em_step)

        print('Classification Repport for validation set:\n', classification_report(y_val_label, q_z_i_val_label, target_names=LABEL_NAMES))
        print('auc_val:', auc_val)
        print('Classification Repport for test set:\n', classification_report(y_test_label, q_z_i_test_label, target_names=LABEL_NAMES))
        print('auc_test:', auc_test)

        print('Classification Repport for validation set (theta):\n', classification_report(y_val_label, theta_i_val_label, target_names=LABEL_NAMES))
        print('auc_val_theta:', auc_val_theta)
        print('Classification Repport for test set (theta):\n', classification_report(y_test_label, theta_i_test_label, target_names=LABEL_NAMES))
        print('auc_test_theta:', auc_test_theta)

    
    if __name__ == '__main__':
        plt.plot(accuracy_theta_i_test, marker='o', label='accuracy_theta_i_test')
        plt.plot(accuracy_theta_i_val, marker='o', label='accuracy_theta_i_val')
        plt.plot(accuracy_q_z_i_test, marker='o', label='accuracy_q_z_i_test')
        plt.plot(accuracy_q_z_i_val, marker='o', label='accuracy_q_z_i_val')
        plt.legend()
        plt.show()
    weights = classifier.get_weights()
    # pd.DataFrame(np.concatenate((column_names[1:], weights[0]), axis=1)).to_csv(weights_after_em, encoding="utf-8")

    report = pd.DataFrame([accuracy_theta_i_test,auc_theta_i_test,accuracy_theta_i_val,accuracy_q_z_i_test,accuracy_q_z_i_val],index=['accuracy_theta_i_test','auc_theta_i_test','accuracy_theta_i_val','accuracy_q_z_i_test','accuracy_q_z_i_val']).transpose()
    report = report.describe()
    return q_z_i, alpha, beta, theta_i, classifier, report


def run(influencer_file_labeled, annotation_file, labels_file, tweet2vec_file, tweet2vec_dim, theta_file,
    evaluation_file, weights_before_em, weights_after_em, total_epochs, n_neurons, steps, supervision_rate,
    iterr, sampling_rate, worker_reliability_file, influencer_quality_file, random_sampling,new_alpha_value,multiple_input):
    tweet2vec = pd.read_csv(tweet2vec_file)

    influencer_labeled = pd.read_csv(influencer_file_labeled, sep=",")

    influencer_labeled = pd.merge(influencer_labeled, tweet2vec[['screen_name','tweet2vec']], how='inner', on=['screen_name'])
    influencer_labeled = influencer_labeled[influencer_labeled['tweet2vec'].notna()]

    labeled_embeddings = []

    for index, row in influencer_labeled.iterrows():
        labeled_embeddings.append(np.fromstring(row['tweet2vec'][1:-1], dtype=float, sep=' '))

    labeled_embeddings = np.array(labeled_embeddings)

    influencer_labeled = influencer_labeled.drop(['screen_name','tweet2vec'], axis=1)

    column_names = np.array(influencer_labeled.columns).reshape((influencer_labeled.shape[1], 1))
    for i in range(0,tweet2vec_dim):
        column_names = np.append(column_names, np.array([['vector' + str(i)]]), axis=0)
    print(column_names.shape)
    annotation_matrix = pd.read_csv(annotation_file, sep=",",header=None)
    labels = pd.read_csv(labels_file, sep=",")

    social_features_labeled = influencer_labeled.values[:,1:]

    # Encode labels
    dummies = pd.get_dummies(labels['label'])
    categories = dummies.columns
    true_labels_pr = dummies.values

    print (influencer_labeled.values[:, [0]].shape,social_features_labeled.shape,true_labels_pr.shape)

    social_features_labeled = np.concatenate(
        (influencer_labeled.values[:, [0]], social_features_labeled, labeled_embeddings, true_labels_pr), axis=1)
    soc_label_bsh = social_features_labeled.copy()

    m = social_features_labeled.shape[1]
    true_labels = social_features_labeled[:, (m - NUMBER_OF_LABELS):]
    social_features_labeled = social_features_labeled[:, :(m - NUMBER_OF_LABELS)]

    n_infls_label = social_features_labeled.shape[0]
    m_feats = social_features_labeled.shape[1]
    n_workers = annotation_matrix[0].unique().shape[0]

    new_order = social_features_labeled[:, 0]

    aij = np.empty((0, 2 + NUMBER_OF_LABELS), int)

    # Encode labels
    dummies = pd.get_dummies(annotation_matrix[2])
    worker_labels = dummies.values

    annotation_matrix = annotation_matrix[[0,1]].values
    annotation_matrix = np.concatenate((annotation_matrix,worker_labels), axis=1)

    for worker in range(0, n_workers):
        worker_aij = annotation_matrix[annotation_matrix[:, 0] == worker]
        worker_aij_s = np.empty((0, 2 + NUMBER_OF_LABELS), int)
        for i in range(0, n_infls_label):
            if worker_aij[worker_aij[:, 1] == new_order[i]].size > 0:
                worker_aij_s = np.concatenate((worker_aij_s, worker_aij[worker_aij[:, 1] == new_order[i]]))
            else:
                no_answer = np.zeros(2 + NUMBER_OF_LABELS, dtype = int)
                no_answer[0] = worker
                no_answer[1] = i
                no_answer = no_answer.reshape(-1,2 + NUMBER_OF_LABELS)
                worker_aij_s = np.concatenate((worker_aij_s,no_answer))
        aij = np.concatenate((aij, worker_aij_s))
    all_workers = np.unique(annotation_matrix[:, 0])

    aij_s = np.empty((0, 2 + NUMBER_OF_LABELS), int)

    for worker in all_workers:
        worker_aij = aij[aij[:, 0] == worker]
        T_w = worker_aij[~np.all(worker_aij[:,2:] == 0, axis=1)]
        T_w_n_all = worker_aij[np.all(worker_aij[:,2:] == 0, axis=1)]
        if int(T_w.shape[0] * sampling_rate) < T_w_n_all.shape[0]:
            indices = random.sample(range(T_w_n_all.shape[0]), int(T_w.shape[0] * sampling_rate))
        else:
            indices = random.sample(range(T_w_n_all.shape[0]), T_w_n_all.shape[0])
        T_w_n = T_w_n_all[indices, :].copy()
        aij_s = np.concatenate((aij_s, T_w, T_w_n))

    if random_sampling:
        T_w_n  = aij_s[np.all(aij_s[:,2:] == 0, axis=1)]
        aij_s = aij_s[~np.all(aij_s[:,2:] == 0, axis=1)]

        num_no_answer = T_w_n.shape[0]
        # equal dist.
        label_ditribution = np.full(( NUMBER_OF_LABELS), 1/NUMBER_OF_LABELS)
        # custom
        # label_ditribution = np.array([1/6,1/6,2/3])

        random_labels = np.empty((0,), int)
        for i in LABEL_INDEX:
            random_labels = np.concatenate((random_labels,np.repeat(i, int(label_ditribution[i]*num_no_answer))))
        random_labels = np.concatenate((random_labels,np.random.randint(3, size=num_no_answer-random_labels.shape[0])))
        np.random.shuffle(random_labels)

        for i, e in enumerate(T_w_n):
            e[2 + random_labels[i]] = 1
        aij_s = np.concatenate((aij_s, T_w_n))

    # size_train = int(supervision_rate * n_infls_label)
    # percentage_train = 0
    # for infl in range(size_train):
    #     infl_idx = social_features_labeled[infl, 0]
    #     infl_aij = annotation_matrix[annotation_matrix[:, 1] == infl_idx]
    #     percentage_train += np.sum(infl_aij[:, 2])

    # print("% of ones in the training=", (percentage_train * 100) / aij_s.shape[0])

    # print(np.sum(aij_s[:, 2]), aij_s.shape[0])
    # print("% of ones in the matrix=", (np.sum(aij_s[:, 2]) * 100) / aij_s.shape[0])


    # with open(evaluation_file, 'a') as file:
    #     file.write("sampling rate," + str(sampling_rate))
    #     file.write('\n')
    #     file.write("nb neurons," + str(n_neurons))
    #     file.write('\n')
    nn_em_in = nn_em()
    print (social_features_labeled.shape, true_labels.shape)

    social_features_labeled = social_features_labeled[:,1:]
    m_feats = m_feats - 1

    q_z_i, alpha, beta, theta_i, classifier, report = var_em(nn_em_in,n_infls_label,aij_s,new_order,n_workers,\
                                                                social_features_labeled,\
                                                                true_labels,supervision_rate, column_names,\
                                                                n_neurons,m_feats,weights_before_em,weights_after_em,\
                                                                iterr,total_epochs,evaluation_file,theta_file,steps,new_alpha_value,multiple_input,tweet2vec_dim)

    report.to_csv(evaluation_file)
    df = pd.read_csv(weights_before_em,
                     names=['name', 'weight']).sort_values(by=['weight'],ascending=False)
    df.to_csv(weights_before_em)
    df = pd.read_csv(weights_after_em,
                     names=['name', 'weight']).sort_values(by=['weight'],ascending=False)
    df.to_csv(weights_after_em)
    worker_reliability = np.concatenate((np.arange(n_workers).reshape(n_workers, 1), alpha, beta), axis=1)
    influencer_quality = np.concatenate(
        (social_features_labeled[:, [0]], true_labels, q_z_i, theta_i), axis=1)
    pd.DataFrame(worker_reliability).to_csv(worker_reliability_file, index=False)
    pd.DataFrame(influencer_quality).to_csv(influencer_quality_file, index=False)
    return report
    

if __name__ == '__main__':
    # load default arguments
    args = arguments.args

    run(**args)