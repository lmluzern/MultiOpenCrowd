# use python3
import var_em
import arguments
import pandas as pd

epochs = 10

# load default arguments
args = arguments.args

# range from 0 to 50 -> with 50+ error
for rate in [0, 0.1, 1] + [x * 10.0 for x in range(1, 6)]:
	args['sampling_rate'] = rate
	args['iterr'] = 20

	report = pd.DataFrame()
	for i in range(epochs):
		# returns performance report
		r = var_em.run(**args)
		if report.empty:
			report = r.copy()
		else:
			report = report.add(r)
	report = report/epochs
	report.to_csv(('../output/exp_sampling_rate/evaluation_sr_' + 
		str(args['sampling_rate']) + '_sup_' + str(args['supervision_rate']) + '.csv'))