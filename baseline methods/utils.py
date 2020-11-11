import csv
import random
import pandas as pd

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

def gete2wlandw2el(annotation_file):
    e2wl = {}
    w2el = {}
    label_set=[]

    aij = pd.read_csv(annotation_file).values

    for e in aij:
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