import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
import ast
import random
import os


# fix random seed
random.seed(1390) 
np.random.seed(round(random.random()*10000)) 


numInstances = 1000
numFolds = 10
numInstancesPerFold = numInstances // numFolds


theta_dim = 45
x_dim = 311
features_dim = theta_dim + 2*x_dim

nonlinear_dim = 124
part_dim = 2*nonlinear_dim


features = np.empty([numInstances, features_dim], dtype=float)
part_points = np.empty([numInstances, part_dim], dtype=float)


base_dir = '../../../'

for inst in range(numInstances):
	instance_file = base_dir + 'data/pooling/pooling_instances/json/random_schweiger_c15_e150_q1_' + str(inst+1) + '.json'
	presolve_file = base_dir + 'results/pooling/presolve_output/presolve_solutions/random_schweiger_c15_e150_q1_' + str(inst+1) + '_presolve.json'
	mccormick_file = base_dir + 'results/pooling/mccormick_output/mccormick_solutions/random_schweiger_c15_e150_q1_' + str(inst+1) + '_mccormick.json'
	part_file = base_dir + 'results/pooling/strong_partitioning_points/random_schweiger_c15_e150_q1_sp/random_schweiger_c15_e150_q1_' + str(inst+1) + '_part_mpbngc.json'


	with open(instance_file) as data_file:
		data_instance = json.load(data_file)
	with open(presolve_file) as data_file2:
		data_presolve = json.load(data_file2)
	with open(mccormick_file) as data_file3:
		data_mccormick = json.load(data_file3)
	with open(part_file) as data_file4:
		data_part = json.load(data_file4)
	
	theta_specs = np.array(data_instance["theta_specs_inputs"]).flatten()
	features[inst][0:theta_dim] = theta_specs

	x_ij_presolve = data_presolve["x_ij_presolve"]
	x_il_presolve = data_presolve["x_il_presolve"]
	x_lj_presolve = data_presolve["x_lj_presolve"]
	q_presolve = data_presolve["q_presolve"]
	presolve_soln = x_ij_presolve + x_il_presolve + x_lj_presolve + q_presolve
	features[inst][theta_dim:theta_dim+x_dim] = presolve_soln

	x_ij_mccormick = data_mccormick["x_ij_mccormick"]
	x_il_mccormick = data_mccormick["x_il_mccormick"]
	x_lj_mccormick = data_mccormick["x_lj_mccormick"]
	q_mccormick = data_mccormick["q_mccormick"]
	mccormick_soln = x_ij_mccormick + x_il_mccormick + x_lj_mccormick + q_mccormick
	features[inst][theta_dim+x_dim:theta_dim+2*x_dim] = mccormick_soln

	part_points[inst] = data_part["part_mpbngc"]



rand_perm = np.random.permutation(numInstances)
shuffled_features = features[rand_perm]
shuffled_part_points = part_points[rand_perm]

predicted_part_points = np.empty([numInstances, part_dim], dtype=float)


for fold in range(numFolds):
	testing_set = list(np.arange(fold*numInstancesPerFold, (fold+1)*numInstancesPerFold))

	if fold == 0:
		training_set = list(np.arange(numInstancesPerFold, numInstances))
	elif fold == numFolds-1:
		training_set = list(np.arange(0, numInstances - numInstancesPerFold))
	else:
		training_set = list(np.arange(0, fold*numInstancesPerFold)) + list(np.arange((fold+1)*numInstancesPerFold, numInstances))

	testing_features = shuffled_features[testing_set]
	testing_part_points = shuffled_part_points[testing_set]

	training_features = shuffled_features[training_set]
	training_part_points = shuffled_part_points[training_set]

	print("")
	print("Running Fold #", fold+1)
	for part_num in range(part_dim):
		regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=25), random_state=0, n_estimators=1000)
		regressor.fit(training_features, training_part_points[:, part_num])
		part_pred = regressor.predict(testing_features)

		predicted_part_points[rand_perm[testing_set], part_num] = part_pred

		mse_val = mean_squared_error(testing_part_points[:, part_num], predicted_part_points[rand_perm[testing_set], part_num])
		mae_val = mean_absolute_error(testing_part_points[:, part_num], predicted_part_points[rand_perm[testing_set], part_num])
		print("part_num: ", part_num, "  test mse: ", mse_val, "  test mae: ", mae_val)
	print("")


	os.makedirs('./ab_predicted_part_points', exist_ok=True)

	test_instances = ""
	for i in rand_perm[testing_set]:
		predicted_part_points_dict = dict()
		predicted_part_points_dict["part_points"] = list(predicted_part_points[i,:])

		part_file = 'ab_predicted_part_points/random_schweiger_c15_e150_q1_' + str(i+1) + '_ab_pred.json'
		with open(part_file, 'w', encoding='utf-8') as outF:
			json.dump(predicted_part_points_dict, outF, ensure_ascii=False, indent=4)

		test_instances += str(i+1) + ","

	print("test instances: ", test_instances)
