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
random.seed(3814) 
np.random.seed(round(random.random()*10000)) 


numInstances = 1000
numFolds = 10
numInstancesPerFold = numInstances // numFolds


theta_dim = 9
x_dim = 10
features_dim = theta_dim + 2*x_dim
part_dim = 2*x_dim


features = np.empty([numInstances, features_dim], dtype=float)
part_points = np.empty([numInstances, part_dim], dtype=float)


base_dir = '../../../../'

for inst in range(numInstances):
	features_file = base_dir + 'results/bilinear/n=10/ml_features/qcqp_v10_b45_s100_' + str(inst+1) + '_features.json'
	with open(features_file) as data_file:
		data_loaded = json.load(data_file)
	
	theta = data_loaded["theta"]
	features[inst][0:theta_dim] = np.concatenate(theta).flat

	x_presolve = data_loaded["x_presolve"]
	features[inst][theta_dim:theta_dim+x_dim] = x_presolve

	x_mccormick = data_loaded["x_mccormick"]
	features[inst][theta_dim+x_dim:theta_dim+2*x_dim] = x_mccormick

	part_points[inst] = data_loaded["part_mpbngc"]


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
	for part_num in range(20):
		regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=25), random_state=0, n_estimators=1000)
		regressor.fit(training_features, training_part_points[:, part_num])
		part_pred = regressor.predict(testing_features)

		predicted_part_points[rand_perm[testing_set], part_num] = part_pred

		mse_val = mean_squared_error(testing_part_points[:, part_num], predicted_part_points[rand_perm[testing_set], part_num])
		mae_val = mean_absolute_error(testing_part_points[:, part_num], predicted_part_points[rand_perm[testing_set], part_num])
		print("part_num: ", part_num, "  mse: ", mse_val, "  mae: ", mae_val)
	print("")


os.makedirs('./ab_predicted_part_points', exist_ok=True)

for i in range(numInstances):
	predicted_part_points_dict = dict()
	predicted_part_points_dict["part_points"] = list(predicted_part_points[i,:])

	part_file = 'ab_predicted_part_points/qcqp_v10_b45_s100_' + str(i+1) + '_ab_pred.json'
	with open(part_file, 'w', encoding='utf-8') as outF:
		json.dump(predicted_part_points_dict, outF, ensure_ascii=False, indent=4)
