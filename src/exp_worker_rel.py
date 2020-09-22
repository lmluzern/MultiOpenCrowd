# use python3
import var_em
import arguments
import pandas as pd

epochs = 10

# load default arguments
args = arguments.args

for i in range(2):
	args['new_alpha_value'] = 0.6
	args['iterr'] = 20
	args['random_sampling'] = False
	if i == 0:
		args['random_sampling'] = True

	report = pd.DataFrame()
	for i in range(epochs):
		# returns performance report
		r = var_em.run(**args)
		if report.empty:
			report = r.copy()
		else:
			report = report.add(r)
	report = report/epochs
	report.to_csv(('../output/exp_worker_rel/evaluation_sr_' + 
		str(args['sampling_rate']) + '_sup_' + str(args['supervision_rate']) + '_rand_samp_' + str(args['random_sampling']) +'.csv'))