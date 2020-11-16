import csv
import random
import pandas as pd
import numpy as np

def gete2t(train_size,truth_labels):
    e2t = {}

    for example in range(train_size):
        e2t[str(example)] = str(truth_labels[example])

    return e2t

def getLabels(e2lpd):
    labels = []

    for e in e2lpd:
        temp = 0
        for label in e2lpd[e]:
            if temp < e2lpd[e][label]:
                temp = e2lpd[e][label]
        
        candidate = []

        for label in e2lpd[e]:
            if temp == e2lpd[e][label]:
                candidate.append(label)

        truth = random.choice(candidate)
        dct = {}
        dct['example'] = int(e)
        dct['label'] = int(truth)
        labels.append(dct)

    return labels

def gete2wlandw2el(annotation_file,sampling_rate):
    e2wl = {}
    w2el = {}
    label_set=[]

    aij = pd.read_csv(annotation_file).values
    all_workers = np.unique(aij[:, 1])
    all_items = np.unique(aij[:, 0])

    # negative sampling
    aij_s = np.empty((0, 3), int)

    for worker in all_workers:
    	worker_aij = aij[aij[:, 1] == worker]
    	aij_s = np.concatenate((aij_s, worker_aij))

    	num_of_answers = worker_aij.shape[0]
    	possible_items = np.delete(all_items, worker_aij[:,0])
    	# print(worker_aij)

    	num_of_samples = int(num_of_answers * sampling_rate)
    	if possible_items.shape[0] < num_of_samples:
    		num_of_samples = possible_items.shape[0]

    	for i in np.random.choice(possible_items,num_of_samples, replace=False):
    		new_answer = np.zeros(3, dtype = int)
    		new_answer[0] = i
    		new_answer[1] = worker
    		new_answer[2] = 2
    		aij_s = np.concatenate((aij_s, new_answer.reshape(-1,3)))
    	
    for e in aij_s:
        example = str(e[0])
        worker = str(e[1])
        label = str(e[2])
        if example not in e2wl:
            e2wl[example] = []
        e2wl[example].append([worker,label])

        if worker not in w2el:
            w2el[worker] = []
        w2el[worker].append([example,label])

        if label not in label_set:
            label_set.append(label)

    return e2wl,w2el,label_set

def gete2wlandw2el_OLD(annotation_file):
    e2wl = {}
    w2el = {}
    label_set=[]

    f = open(annotation_file, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        example, worker, label = line
        if example not in e2wl:
            e2wl[example] = []
        e2wl[example].append([worker,label])

        if worker not in w2el:
            w2el[worker] = []
        w2el[worker].append([example,label])

        if label not in label_set:
            label_set.append(label)

    return e2wl,w2el,label_set
