# use python3
import var_em
import arguments
import pandas as pd

epochs = 10

# load default arguments
args = arguments.args

for rate in [x * 0.1 for x in range(0, 11)]:
	args['new_alpha_value'] = rate
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
	report.to_csv(('../output/exp_new_alpha_value/evaluation_sr_' + 
		str(args['sampling_rate']) + '_sup_' + str(args['supervision_rate']) + '_new_alpha_' + str(args['new_alpha_value']) +'.csv'))