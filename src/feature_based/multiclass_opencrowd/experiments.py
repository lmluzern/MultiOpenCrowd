# use python3
from feature_based.multiclass_opencrowd import var_em
from feature_based.multiclass_opencrowd import arguments
import pandas as pd


def run_experiment(epochs, file_out, value_range, value_name, random_sampling, sampling_rate,
                   supervision_rate, iterr, file_labeled, annotation_file, labels_file, tweet2vec_file):
    # load default arguments
    args = arguments.args
    args['random_sampling'] = random_sampling
    args['sampling_rate'] = sampling_rate
    args['supervision_rate'] = supervision_rate
    args['iterr'] = iterr
    args['influencer_file_labeled'] = file_labeled
    args['annotation_file'] = annotation_file
    args['labels_file'] = labels_file
    args['tweet2vec_file'] = tweet2vec_file

    out = pd.DataFrame()
    for value in value_range:
        args[value_name] = value
        report = pd.DataFrame()
        for i in range(epochs):
            # returns performance report
            r = var_em.run(**args)
            if report.empty:
                report = r.copy()
            else:
                report = report.add(r)
        report = report / epochs
        report['iterr'] = args['iterr']
        report['sampling_rate'] = args['sampling_rate']
        report['supervision_rate'] = args['supervision_rate']
        report['new_alpha_value'] = args['new_alpha_value']
        report['random_sampling'] = args['random_sampling']
        report['epochs'] = epochs
        if out.empty:
            out = report.copy()
        else:
            out = out.append(report, ignore_index=True)
    out.to_csv(file_out)
    return out
