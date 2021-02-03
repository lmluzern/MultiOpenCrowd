# use python3
import var_em
import arguments
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


# run_experiment(epochs=10, file_out='../output/exp_supervision_rate.csv', value_range=[
#                0.3, 0.4, 0.5, 0.6, 0.7, 0.8], value_name='supervision_rate', random_sampling=False, sampling_rate=6, supervision_rate=0, iterr=5, file_labeled='../input/influencer_labeled.csv', annotation_file='../input/influencer_aij.csv', labels_file='../input/influencer_labels.csv', tweet2vec_file='../input/influencer_tweet2vec_glove_200d.csv')
# run_experiment(epochs=10, file_out='../output/exp_sampling_rate.csv', value_range=[
#                1,2,3,4,5,6,7,8,9,10], value_name='sampling_rate', random_sampling=False, sampling_rate=0, supervision_rate=0.6, iterr=5, file_labeled='../input/influencer_labeled.csv', annotation_file='../input/influencer_aij.csv', labels_file='../input/influencer_labels.csv', tweet2vec_file='../input/influencer_tweet2vec_glove_200d.csv')

# run_experiment(epochs=10, file_out='../output/exp_supervision_rate.csv', value_range=[
#                0.3, 0.4, 0.5, 0.6, 0.7, 0.8], value_name='supervision_rate', random_sampling=False, sampling_rate=0.2, supervision_rate=0, iterr=10, file_labeled='../input/sentiment_labeled.csv', annotation_file='../input/sentiment_aij.csv', labels_file='../input/sentiment_labels.csv', tweet2vec_file='../input/sentiment_tweet2vec_glove_200d.csv')

# run_experiment(epochs=10, file_out='../output/exp_supervision_rate.csv', value_range=[
#                0.3, 0.4, 0.5, 0.6, 0.7, 0.8], value_name='supervision_rate', random_sampling=False, sampling_rate=4, supervision_rate=0, iterr=20, file_labeled='../input/sentiment_labeled.csv', annotation_file='../input/sentiment_sparse_aij.csv', labels_file='../input/sentiment_labels.csv', tweet2vec_file='../input/sentiment_tweet2vec_glove_200d.csv')
