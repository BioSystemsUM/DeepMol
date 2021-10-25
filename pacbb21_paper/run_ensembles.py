import argparse
import os
from datetime import datetime
import pickle

import tensorflow as tf

from model_build_functions import BUILDERS
from src.loaders.Loaders import CSVLoader
from src.metrics.Metrics import Metric
from src.metrics.metricsFunctions import r2_score, mean_squared_error, pearson_score, accuracy_score, roc_auc_score, \
	prc_auc_score, precision_score, recall_score, spearman_score
from src.models.kerasModels import KerasModel
from src.models.ensembles import VotingEnsemble
from utils import get_featurizer, save_evaluation_results


def main(dataset_dir, output_filepath, model_names, hyperparams, gpu):
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu
	os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

	output_dir = os.path.join('..', 'pacbb21_paper', 'results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	os.makedirs(output_dir)

	# Load Dataset
	print('Loading data')
	dataset_name = os.path.split(dataset_dir)[-1]
	train_filepath = os.path.join(dataset_dir, 'train_' + dataset_name + '.csv')
	test_filepath = os.path.join(dataset_dir, 'test_' + dataset_name + '.csv')

	train_dataset = CSVLoader(dataset_path=train_filepath,
	                          mols_field='mols',
	                          labels_fields='y')
	train_dataset = train_dataset.create_dataset()
	train_dataset.get_shape()
	test_dataset = CSVLoader(dataset_path=test_filepath,
	                         mols_field='mols',
	                         labels_fields='y')
	test_dataset = test_dataset.create_dataset()
	test_dataset.get_shape()

	# Define type of prediction task and define scoring metrics accordingly
	if len(set(train_dataset.y)) > 2:
		mode = 'regression'
		metrics = [Metric(mean_squared_error, n_tasks=1),
		           Metric(r2_score, n_tasks=1),
		           Metric(pearson_score, mode='regression', n_tasks=1),
		           Metric(spearman_score, mode='regression', n_tasks=1)]
	else:
		mode = 'classification'
		metrics = [Metric(roc_auc_score, n_tasks=1),
		           Metric(prc_auc_score, n_tasks=1),
		           Metric(accuracy_score, n_tasks=1),
		           Metric(precision_score, n_tasks=1),
		           Metric(recall_score, n_tasks=1)]

	# Load hyperparam values
	with open(hyperparams, 'rb') as f:
		hyperparam_vals = pickle.load(f)

	# Get build functions and featurizers
	print('Getting model build functions and featurizers')
	models = []
	featurizers = []
	for model_name in model_names:
		featurizers.append(get_featurizer(model_name))
		builder = BUILDERS[model_name]
		hyperparams = hyperparam_vals[model_name]
		print(model_name)

		if model_name in ['GraphConv', 'TextCNN', 'Weave', 'MPNN', 'GCN', 'GAT', 'AttentiveFP', 'TorchMPNN']:
			models.append(builder(**hyperparams))
		else:
			models.append(KerasModel(model_builder=builder, mode=mode, **hyperparams))
	hyperparam_vals = {x: hyperparam_vals[x] for x in model_names}

	# Build ensemble
	print('Building the ensemble')
	ensemble = VotingEnsemble(base_models=models, featurizers=featurizers, mode=mode, model_dir=output_dir)

	# Fit ensemble model
	print('Fitting ensemble')
	ensemble.fit(train_dataset)

	# Evaluate ensemble model
	print('Evaluating ensemble on test set')
	train_scores = ensemble.evaluate(train_dataset, metrics)
	test_scores = ensemble.evaluate(test_dataset, metrics)
	if output_filepath is None:
		output_filepath = os.path.join('..', 'pacbb21_paper', 'results', '%s_evaluation_results.csv' % dataset_name)
	save_evaluation_results(test_scores, train_scores, hyperparam_vals, 'VotingEnsemble_%s' % dataset_name, model_dir=output_dir,
	                        output_filepath=output_filepath)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train and evaluate user-defined models')
	parser.add_argument('-i',
	                    '--dataset-dir',
	                    type=str,
	                    help='Path to the directory where the train and test sets are stored for a particular dataset')
	parser.add_argument('-o',
	                    '--output-filepath',
	                    type=str,
	                    help='Results filename')
	parser.add_argument('--model-names',
	                    nargs='+',
	                    type=str,
	                    help='The type of model')
	parser.add_argument('-p',
	                    '--hyperparams',
	                    type=str,
	                    help='Path to file containing the hyperparameter values to use')
	parser.add_argument('-g',
	                    '--gpu',
	                    type=str,
	                    help='The GPU to use')
	args = vars(parser.parse_args())
	print(args)
	main(**args)
