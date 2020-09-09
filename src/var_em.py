import pandas as pd
import csv
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from math import floor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.special import digamma
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
from nn_em import nn_em
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn import metrics
import argparse

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


def update(a, b,n_update,change):
    n_update += 1
    change += np.abs(a - b).sum()/a.shape[0]


    return n_update,change

def optimize_rj(x_train, n_neurons, nb_layers, training_epochs, display_step, batch_size, n_input, alpha, beta):
    graph1 = tf.Graph()
    with graph1.as_default():
        tf.set_random_seed(1)
        # input layer
        x = tf.placeholder(tf.float32, [None, n_input])
        keep_prob = tf.placeholder(tf.float32)
        layer = x
        # hideen layers
        for _ in range(nb_layers):
            layer = tf.layers.dense(inputs=layer, units=n_neurons, activation=tf.nn.tanh)
        # output layer
        alpha_prime = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.relu(x) + 1)
        beta_prime = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.relu(x) + 1)

        dist = tf.distributions.Beta(alpha_prime, beta_prime)
        target_dist = tf.distributions.Beta(alpha, beta)

        loss = tf.distributions.kl_divergence(target_dist, dist)
        cost = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    with tf.Session(graph=graph1) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(x_train) / batch_size)
            x_batches = np.array_split(x_train, total_batch)
            for i in range(total_batch):
                batch_x = x_batches[i]
                _, c = sess.run([optimizer, cost],
                                feed_dict={
                                    x: batch_x,
                                    keep_prob: 0.8
                                })
                avg_cost += c / total_batch
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                      "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        alpha_prime_res, beta_prime_res = sess.run([alpha_prime, beta_prime],
                                                   feed_dict={
                                                       x: x_train,
                                                       keep_prob: 0.8
                                                   })
        print("alpha_prime_res=", alpha_prime_res, "beta_prime_res=", beta_prime_res)
        return alpha_prime_res, beta_prime_res


def e_step(y_train, n_workers, q_z_i, annotation_matrix, alpha, beta, theta_i,true_labels,new_order, max_it=100):
    for it in range(max_it):
        change = 0
        n_update = 0
        # update q(z)

        for infl in new_order.tolist():
            index_infl = np.where(new_order == infl)[0][0]
            assert infl == index_infl
            updated_q_z_i = theta_i[index_infl].copy()
            infl_aij = annotation_matrix[annotation_matrix[:, 1] == infl]
            worker_answers = infl_aij[~np.all(infl_aij[:,2:] == 0, axis=1)]
            T_i = worker_answers[:, 0]
            for worker in T_i.astype(int):
                #worker_id = np.where(all_workers == worker)
                w_answer = worker_answers[worker_answers[:, 0] == worker][:, 2:]
                w_answer_i = np.where(w_answer[0] == 1)[0][0]
                alpha_val = alpha[worker]
                beta_val =  beta[worker]

                updated_q_z_i[w_answer_i] = updated_q_z_i[w_answer_i] * np.exp(digamma(alpha_val) - digamma(alpha_val + beta_val))

                for no_answer_i in np.delete(LABEL_INDEX,w_answer_i):
                    updated_q_z_i[no_answer_i] = updated_q_z_i[no_answer_i] * np.exp(digamma(beta_val) - digamma(alpha_val + beta_val))

            # normalize
            new_q_z_i = updated_q_z_i * 1.0 / (updated_q_z_i.sum())
            n_update, change = update(q_z_i[index_infl], new_q_z_i,n_update,change)
            q_z_i[index_infl] = new_q_z_i

        q_z_i = np.concatenate((y_train, q_z_i[y_train.shape[0]:]))

        # update q(r)
        new_alpha = np.zeros((n_workers, 1))
        new_beta = np.zeros((n_workers, 1))
        for worker in range(0, n_workers):
            new_alpha[worker] = alpha[worker]
            new_beta[worker] = beta[worker]

        for worker in range(0, n_workers):
            worker_aij = annotation_matrix[annotation_matrix[:, 0] == worker]
            # T_j_1 = worker_aij[worker_aij[:,2] == 1][:, 1]
            T_j = worker_aij[~np.all(worker_aij[:,2:] == 0, axis=1)]
            for infl in T_j[:, 1].astype(int):
                if (np.where(new_order == infl)[0].shape[0]) > 0:
                    index_infl = np.where(new_order == infl)[0][0]
                    worker_answer = T_j[T_j[:, 1] == infl][:, 2:]
                    worker_answer_i = np.where(worker_answer[0] == 1)[0][0]
                    assert infl == index_infl
                    new_alpha[worker] += q_z_i[index_infl][worker_answer_i]
                    new_beta[worker] += 1 - q_z_i[index_infl][worker_answer_i]
                else:
                    assert 1==0

        for worker in range(0, n_workers):
            n_update, change = update(alpha[worker], new_alpha[worker],n_update,change)
            alpha[worker] = new_alpha[worker]
            n_update, change = update(beta[worker], new_beta[worker],n_update,change)
            beta[worker] = new_beta[worker]
        avg_change = change * 1.0 / n_update

        if avg_change < 0.01:
            break

    return q_z_i,alpha,beta

def m_step(nn_em,q_z_i, classifier, social_features, total_epochs, steps, y_test, y_val,start_val,alpha, beta):
    theta_i, classifier, weights = nn_em.train_m_step(classifier, social_features,
                                                      q_z_i,
                                                      steps, total_epochs, y_test, y_val,start_val)

    return theta_i,classifier


def var_em(nn_em_in, n_infls_label,aij_s,new_order, n_workers, social_features_labeled, true_labels, supervision_rate, \
           column_names, n_neurons, hidden, m_feats, weights_before_em,weights_after_em,iterr,total_epochs,evaluation_file,theta_file,steps,nb_hidden_layer):
    n_infls = n_infls_label
    q_z_i, A, B = init_probabilities(n_infls)

    alpha, beta = init_alpha_beta(A, B, n_workers)

    X_train, X_test, y_train, y_test = train_test_split(social_features_labeled, true_labels,
                                                        test_size=(1 - supervision_rate), shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)

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

    while (LA.norm(theta_i - old_theta_i) > epsilon) and (steps_it0 < total_epochs):
        old_theta_i = theta_i.copy()
        classifier.fit(X_train, y_train, epochs=steps, verbose=0)
        theta_i_val = classifier.predict(X_val)
        theta_i_test = classifier.predict(X_test)

        theta_i_val_label = np.argmax(theta_i_val,axis=1)
        theta_i_test_label = np.argmax(theta_i_test,axis=1) 

        theta_i = np.concatenate((y_train, theta_i_val, theta_i_test))
        eval_model_test = accuracy_score(y_test_label, theta_i_test_label)
        eval_model_val = accuracy_score(y_val_label, theta_i_val_label)
        if steps_it0 % 10 == 0:
            print("epoch", steps_it0, " convergence:", LA.norm(theta_i - old_theta_i), \
                "val", eval_model_val, "test", eval_model_test)
            # print('val:')
            # print(classification_report(y_val_label, theta_i_val_label, target_names=LABEL_NAMES))
            # print('test:')
            # print(classification_report(y_test_label, theta_i_test_label, target_names=LABEL_NAMES))
        steps_it0 += 1

    weights = classifier.get_weights()
    pd.DataFrame(np.concatenate((column_names[1:], weights[0]), axis=1)).to_csv(weights_before_em, encoding="utf-8")

    start_val = X_train.shape[0]
    end_val = X_train.shape[0] + X_val.shape[0]

    auc_val = roc_auc_score(y_val, theta_i_val,multi_class="ovo",average="macro")
    auc_test = roc_auc_score(y_test, theta_i_test,multi_class="ovo",average="macro")

    print('Classification Repport for validation set:\n', classification_report(y_val_label, theta_i_val_label, target_names=LABEL_NAMES))
    print('auc_val:', auc_val)
    print('Classification Repport for test set:\n', classification_report(y_test_label, theta_i_test_label, target_names=LABEL_NAMES))
    print('auc_test:', auc_test)

    theta_i = np.concatenate((y_train, theta_i_val, theta_i_test))
    theta_quality = np.concatenate((true_labels, theta_i), axis=1)
    pd.DataFrame(theta_quality).to_csv(theta_file, index=False)

    social_features = social_features_labeled
    
    em_step = 0
    while em_step < iterr:
        # variational E step
        q_z_i, alpha, beta = e_step(y_train, n_workers, q_z_i, aij_s, alpha,
                                               beta, theta_i, true_labels,new_order)

        # variational M step
        theta_i, classifier = m_step(nn_em_in, q_z_i, classifier, social_features, total_epochs, steps, y_test, y_val,
                                     start_val, alpha, beta)
        em_step += 1

        q_z_i_val_label = np.argmax(q_z_i[start_val:end_val],axis=1)
        q_z_i_test_label = np.argmax(q_z_i[end_val:],axis=1)

        auc_val = roc_auc_score(y_val, q_z_i[start_val:end_val],multi_class="ovo",average="macro")
        auc_test = roc_auc_score(y_test, q_z_i[end_val:],multi_class="ovo",average="macro")

        theta_i_val_label = np.argmax(theta_i[start_val:end_val],axis=1)
        theta_i_test_label = np.argmax(theta_i[end_val:],axis=1)

        auc_val_theta = roc_auc_score(y_val, theta_i[start_val:end_val],multi_class="ovo",average="macro")
        auc_test_theta = roc_auc_score(y_test_label, theta_i[end_val:],multi_class="ovo",average="macro")

        print('Classification Repport for validation set:\n', classification_report(y_val_label, q_z_i_val_label, target_names=LABEL_NAMES))
        print('auc_val:', auc_val)
        print('Classification Repport for test set:\n', classification_report(y_test_label, q_z_i_test_label, target_names=LABEL_NAMES))
        print('auc_test:', auc_test)

        print('Classification Repport for validation set (theta):\n', classification_report(y_val_label, theta_i_val_label, target_names=LABEL_NAMES))
        print('auc_val_theta:', auc_val_theta)
        print('Classification Repport for test set (theta):\n', classification_report(y_test_label, theta_i_test_label, target_names=LABEL_NAMES))
        print('auc_test_theta:', auc_test_theta)

    weights = classifier.get_weights()
    pd.DataFrame(np.concatenate((column_names[1:], weights[0]), axis=1)).to_csv(weights_after_em, encoding="utf-8")
    return q_z_i, alpha, beta, theta_i, classifier

def parse_args():
    parser = argparse.ArgumentParser(
        description="EM method")
    parser.add_argument("--labeled_social_features",
                        type=str,
                        required=True,
                        help="inputfile labeled social features")

    parser.add_argument("--unlabeled_social_features",
                        type=str,
                        required=True,
                        help="inputfile unlabeled social features")

    parser.add_argument("--annotation_matrix",
                        type=str,
                        required=True,
                        help="inputfile of the annotation matrix")

    parser.add_argument("--labels",
                        type=str,
                        required=True,
                        help="inputfile of labels")

    parser.add_argument("--total_epochs_nn",
                        default=10,
                        type=int,
                        help="number of epochs for the Neural network at the M step")

    parser.add_argument("--total_neurons_nn",
                        default=10,
                        type=int,
                        help="number of neurons for the Neural network at the M step")

    parser.add_argument("--nb_hidden_layer",
                        default=1,
                        type=int,
                        help="number of hidden layer for the Neural network at the M step")

    parser.add_argument("--steps",
                        default=1,
                        type=int,
                        help="number of steps for the Neural network at the M step")

    parser.add_argument("--hidden_layer",
                        default=False,
                        type=bool,
                        help="use hidden layer in the NN")

    parser.add_argument("--supervision_rate",
                        default=0.6,
                        type=float,
                        help="how much to use for training")

    parser.add_argument("--sampling_rate",
                        default=1.0,
                        type=float,
                        help="how much to use for negative sampling for the e step")

    parser.add_argument("--nb_iterations_EM",
                        default=10,
                        type=int,
                        help="number of iterations for the EM")

    parser.add_argument("--worker_reliability_file",
                        type=str,
                        required=True,
                        help="worker reliability file output of the model")

    parser.add_argument("--influencer_quality_file",
                        type=str,
                        required=True,
                        help="influencer quality file output of the model")

    parser.add_argument("--evaluation_file",
                        type=str,
                        required=True,
                        help="evaluation result after each iteration")

    parser.add_argument("--theta_file",
                        type=str,
                        required=True,
                        help="theta result after nn")

    parser.add_argument("--weights_before_em",
                        type=str,
                        required=True,
                        help="inputfile LR")

    parser.add_argument("--weights_after_em",
                        type=str,
                        required=True,
                        help="inputfile weights EM")

    parser.add_argument("--tweet2vec",
                        type=str,
                        required=True,
                        help="inputfile tweet2vec")

    parser.add_argument("--tweet2vec_dim",
                        type=int,
                        help="dimension of tweet2vec")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    influencer_file_labeled = args.labeled_social_features #'../input/simple_example_vem_labeled.csv'
    influencer_file_unlabeled = args.unlabeled_social_features # '../input/simple_example_vem_unlabeled.csv'
    annotation_file = args.annotation_matrix #'../input/aij_simple_example_vem.csv'
    labels_file = args.labels #'../input/simple_example_vem_labels.csv'
    tweet2vec_file = args.tweet2vec
    tweet2vec_dim = args.tweet2vec_dim
    theta_file = args.theta_file # '../output/theta_i_vem_se.csv'
    evaluation_file = args.evaluation_file # '../output/evaluation_vem_se.csv'
    weights_before_em = args.weights_before_em #'../output/weights_before_em.csv'
    weights_after_em = args.weights_after_em #'../output/weights_after_em.csv'
    nb_hidden_layer = args.nb_hidden_layer

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
    # annotation_matrix = np.loadtxt(annotation_file, delimiter=',')
    labels = pd.read_csv(labels_file, sep=",")

    # social_features_labeled = preprocessing.scale(influencer_labeled.values[:, 1:])
    social_features_labeled = influencer_labeled.values[:,1:]

    # Encode labels
    dummies = pd.get_dummies(labels['label'])
    categories = dummies.columns
    true_labels_pr = dummies.values

    print (influencer_labeled.values[:, [0]].shape,social_features_labeled.shape,true_labels_pr.shape)

    social_features_labeled = np.concatenate(
        (influencer_labeled.values[:, [0]], social_features_labeled, labeled_embeddings, true_labels_pr), axis=1)
    soc_label_bsh = social_features_labeled.copy()
    #np.random.shuffle(social_features_labeled)

    m = social_features_labeled.shape[1]
    true_labels = social_features_labeled[:, (m - NUMBER_OF_LABELS):]
    social_features_labeled = social_features_labeled[:, :(m - NUMBER_OF_LABELS)]

    n_infls_label = social_features_labeled.shape[0]
    m_feats = social_features_labeled.shape[1]
    # n_workers = np.unique(annotation_matrix[:, 0]).shape[0]
    n_workers = annotation_matrix[0].unique().shape[0]

    new_order = social_features_labeled[:, 0]

    total_epochs = args.total_epochs_nn#10
    n_neurons = args.total_neurons_nn #3
    hidden = args.hidden_layer #false
    steps = args.steps #1
    supervision_rate = args.supervision_rate #0.6
    iterr = args.nb_iterations_EM #10
    sampling_rate = args.sampling_rate #2.0

    aij = np.empty((0, 2 + NUMBER_OF_LABELS), int)

    # Encode labels
    dummies = pd.get_dummies(annotation_matrix[2])
    worker_labels = dummies.values

    annotation_matrix = annotation_matrix[[0,1]].values
    annotation_matrix = np.concatenate((annotation_matrix,worker_labels), axis=1)

    for worker in range(0, n_workers):
        worker_aij = annotation_matrix[annotation_matrix[:, 0] == worker]
        # worker_aij_s = worker_aij.copy()
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
    aij_s = aij

    # for worker in all_workers:
    #     worker_aij = annotation_matrix[annotation_matrix[:, 0] == worker]
    #     T_w = worker_aij[worker_aij[:, 2] == 1]
    #     T_w_n_all = worker_aij[worker_aij[:, 2] == 0]
    #     if int(T_w.shape[0] * sampling_rate) < T_w_n_all.shape[0]:
    #         indices = random.sample(range(T_w_n_all.shape[0]), int(T_w.shape[0] * sampling_rate))
    #     else:
    #         indices = random.sample(range(T_w_n_all.shape[0]), T_w_n_all.shape[0])
    #     T_w_n = T_w_n_all[indices, :]
    #     aij_s = np.concatenate((aij_s, T_w, T_w_n))


    # size_train = int(supervision_rate * n_infls_label)
    # percentage_train = 0
    # for infl in range(size_train):
    #     infl_idx = social_features_labeled[infl, 0]
    #     infl_aij = annotation_matrix[annotation_matrix[:, 1] == infl_idx]
    #     percentage_train += np.sum(infl_aij[:, 2])

    # print("% of ones in the training=", (percentage_train * 100) / aij_s.shape[0])

    # print(np.sum(aij_s[:, 2]), aij_s.shape[0])
    # print("% of ones in the matrix=", (np.sum(aij_s[:, 2]) * 100) / aij_s.shape[0])


    with open(evaluation_file, 'a') as file:
        file.write("sampling rate," + str(sampling_rate))
        file.write('\n')
        file.write("hidden," + str(hidden))
        file.write('\n')
        file.write("nb layers," + str(nb_hidden_layer))
        file.write('\n')
        file.write("nb neurons," + str(n_neurons))
        file.write('\n')
    nn_em_in = nn_em()
    print (social_features_labeled.shape, true_labels.shape)

    social_features_labeled = social_features_labeled[:,1:]
    m_feats = m_feats - 1

    q_z_i, alpha, beta, theta_i, classifier = var_em(nn_em_in,n_infls_label,aij_s,new_order,n_workers,\
                                                                social_features_labeled,\
                                                                true_labels,supervision_rate, column_names,\
                                                                n_neurons,hidden,m_feats,weights_before_em,weights_after_em,\
                                                                iterr,total_epochs,evaluation_file,theta_file,steps,nb_hidden_layer)

    df = pd.read_csv(weights_before_em,
                     names=['name', 'weight']).sort_values(by=['weight'],ascending=False)
    df.to_csv(weights_before_em)
    df = pd.read_csv(weights_after_em,
                     names=['name', 'weight']).sort_values(by=['weight'],ascending=False)
    df.to_csv(weights_after_em)
    worker_reliability_file = args.worker_reliability_file
    influencer_quality_file = args.influencer_quality_file
    worker_reliability = np.concatenate((np.arange(n_workers).reshape(n_workers, 1), alpha, beta), axis=1)
    influencer_quality = np.concatenate(
        (social_features_labeled[:, [0]], true_labels, q_z_i, theta_i), axis=1)
    pd.DataFrame(worker_reliability).to_csv(worker_reliability_file, index=False)
    pd.DataFrame(influencer_quality).to_csv(influencer_quality_file, index=False)
        # print(pd.DataFrame(data=np.concatenate([np.where(q_z_i_0 > q_z_i_0.mean(), 0, 1), true_labels], axis=1),
        #                    columns=['classification', 'truth']))
# Execute main() function