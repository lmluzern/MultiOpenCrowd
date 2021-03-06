# inspired by https://kieranrcampbell.github.io/blog/2016/05/15/gibbs-sampling-bayesian-linear-regression.html (18.11.2020)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import time
from numpy import linalg as LA
import matplotlib.pyplot as plt

def getItemDict(annotation_matrix):
    J = {}
    a_j = {}
    for i in np.unique(annotation_matrix[:,1]):
        J[i] = annotation_matrix[annotation_matrix[:,1] == i][:,0]
        a_j[i] = annotation_matrix[annotation_matrix[:,1] == i][:,2].reshape((-1,1))
    return J,a_j

def getWorkerDict(annotation_matrix):
    I = {}
    a_i = {}
    for j in np.unique(annotation_matrix[:,0]):
        I[j]  = annotation_matrix[annotation_matrix[:,0] == j][:,1]
        a_i[j] = annotation_matrix[annotation_matrix[:,0] == j][:,2].reshape((-1,1))
    return I,a_i

def neg_sampling(aij,sampling_rate):
    all_workers = np.unique(aij[:, 0])
    all_items = np.unique(aij[:, 1])
    num_of_labels = np.unique(aij[:, 2]).shape[0]
    aij_s = np.empty((0, 3), int)

    for worker in all_workers:
        worker_aij = aij[aij[:, 0] == worker]
        aij_s = np.concatenate((aij_s, worker_aij))

        num_of_answers = worker_aij.shape[0]
        possible_items = np.delete(all_items, worker_aij[:,1])

        num_of_samples = int(num_of_answers * sampling_rate)
        if possible_items.shape[0] < num_of_samples:
            num_of_samples = possible_items.shape[0]

        label_ditribution = np.full((num_of_labels), 1/num_of_labels)

        random_labels = np.empty((0,), int)
        for i in range(num_of_labels):
            random_labels = np.concatenate((random_labels,np.repeat(i+1, int(label_ditribution[i]*num_of_samples))))
        random_labels = np.concatenate((random_labels,np.random.randint(1,4, size=num_of_samples-random_labels.shape[0])))
        np.random.shuffle(random_labels)

        j = 0
        for i in np.random.choice(possible_items,num_of_samples, replace=False):
            new_answer = np.zeros(3, dtype = int)
            new_answer[0] = worker
            new_answer[1] = i
            new_answer[2] = random_labels[j]
            # option fixed class
            # new_answer[2] = 2
            j+=1
            aij_s = np.concatenate((aij_s, new_answer.reshape(-1,3)))

    return aij_s

def sample_z_i(gamma_0,mu_0,r_j,a_ij):
    precision = r_j.sum() + gamma_0
    mean = ((r_j*a_ij).sum() + gamma_0 * mu_0)/precision
    return np.random.normal(mean,1/np.sqrt(precision)),1/np.sqrt(precision),mean

def sample_r_j(A_0,B_0,z_i,a_ij):
    A_new = A_0 + a_ij.shape[0]/2.0
    B_new = B_0 + ((a_ij-z_i)**2).sum()/2
    return np.random.gamma(A_new,(1/B_new)),A_new,1/B_new

def gibbs(param,e2t):
    annotation_matrix = pd.read_csv(param['annotation_file'],sep=",",header=None)
    annotation_matrix['label_code'] = pd.factorize(annotation_matrix[2],sort=True)[0] + 1
    annotation_matrix = annotation_matrix.values[:,[0,1,3]].astype(np.int)

    lower_bound = np.min(annotation_matrix[:,2]) - 0.5
    upper_bound = np.max(annotation_matrix[:,2]) + 0.5

    all_workers = np.unique(annotation_matrix[:,0])
    all_items = np.unique(annotation_matrix[:,1])

    # init r with mean A_0/B_0
    r = np.full((all_workers.shape[0],1),param['A_0']/param['B_0'],dtype=float)
    # r = np.random.gamma(param['A_0'],(1/param['B_0']),all_workers.shape[0]).reshape(-1,1)

    z = np.full((all_items.shape[0],1),0,dtype=float)
    trace = np.zeros((param['iters'],all_items.shape[0]))

    trace_precision = np.full((all_items.shape[0],1),0,dtype=float)
    trace_mean = np.full((all_items.shape[0],1),0,dtype=float)
    trace_A = np.full((all_workers.shape[0],1),0,dtype=float)
    trace_B = np.full((all_workers.shape[0],1),0,dtype=float)

    trace_conv = np.full((param['iters'],6),0,dtype=float)

    # annotation_matrix = neg_sampling(annotation_matrix,param['sampling_rate'])

    J,a_j = getItemDict(annotation_matrix)
    I,a_i = getWorkerDict(annotation_matrix)

    for it in range(param['iters']):
        prev_trace_precision = trace_precision.copy()
        prev_trace_mean = trace_mean.copy()
        prev_trace_z = z.copy()
        prev_trace_A = trace_A.copy()
        prev_trace_B = trace_B.copy()
        prev_trace_r = r.copy()
        # for each item label
        for i in all_items:
            # supervision
            if i in e2t:
                z[i] = e2t[i]
                continue
            r_j = r[J[i]]
            a_ij = a_j[i]
            z[i],trace_precision[i],trace_mean[i] = sample_z_i(param['gamma_0'],param['mu_0'],r_j,a_ij)
            while z[i] <= lower_bound or z[i] >= upper_bound:
                z[i],trace_precision[i],trace_mean[i] = sample_z_i(param['gamma_0'],param['mu_0'],r_j,a_ij)
        trace[it,:] = z.transpose()[0].copy()
        
        # for each worker reliability
        for j in all_workers:
            z_i = z[I[j]]
            a_ij = a_i[j]
            r[j],trace_A[j],trace_B[j] = sample_r_j(param['A_0'],param['B_0'],z_i,a_ij)

        trace_conv[it] = LA.norm(trace_precision-prev_trace_precision), LA.norm(trace_mean-prev_trace_mean), LA.norm(trace_A-prev_trace_A), LA.norm(trace_B-prev_trace_B), LA.norm(z-prev_trace_z), LA.norm(r-prev_trace_r)
    # print('r max/mix:', np.max(r), np.min(r))
    return trace,trace_conv

def run(param,e2t):
    start_t = time.time()
    trace,trace_conv = gibbs(param,e2t)
    z_median = np.median(trace[int(param['iters']*param['burn_in_rate']):],axis=0)

    if __name__ == '__main__':
        plt.plot(trace_conv[:,0],label='precision')
        plt.plot(trace_conv[:,1],label='mean')
        plt.plot(trace_conv[:,2],label='A_new')
        plt.plot(trace_conv[:,3],label='B_new')
        plt.plot(trace_conv[:,4],label='z')
        plt.plot(trace_conv[:,5],label='r')
        plt.legend(loc="upper right")
        plt.show()

    return z_median, trace_conv


if __name__ == '__main__':
    param = {
    'annotation_file' : '../input/influencer_aij.csv',
    'labels_file' : '../input/influencer_labels.csv',
    'A_0' : 1,
    'B_0' : 1,
    'gamma_0' : 0.8,
    'mu_0' : 3.5,
    'iters' : 1000,
    'burn_in_rate' : 0.2,
    'supervision_rate' : 0.0,
    'sampling_rate' : 0.0
    }

    z_median, trace_conv = run(param,{})
    ground_truth = pd.factorize(pd.read_csv(param['labels_file'],sep=",")['label'],sort=True)[0] + 1
    accuracy = accuracy_score(ground_truth,z_median.round())
    print('accuracy',accuracy)
