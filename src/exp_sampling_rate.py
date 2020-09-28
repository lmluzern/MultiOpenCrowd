# use python3
import var_em
import arguments
import pandas as pd

epochs = 10

# load default arguments
args = arguments.args

# range from 0 to 50 -> with 50+ error
for rate in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
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