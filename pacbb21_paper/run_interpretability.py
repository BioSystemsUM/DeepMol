import argparse
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from model_build_functions import BUILDERS
from src.loaders.Loaders import CSVLoader
from src.models.kerasModels import KerasModel
from src.featureImportance.shapValues import ShapValues
from src.utils.utils import load_from_disk, normalize_labels_shape
from utils import get_featurizer


def main(dataset_dir, model_name, saved_model, hyperparams, gpu):
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu
	print(os.environ['CUDA_VISIBLE_DEVICES'])
	os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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

	output_dir = os.path.join('..', 'pacbb21_paper', 'results', 'shap', dataset_name)
	os.makedirs(output_dir)

	# Featurization
	print('Featurizing molecules')
	featurizer = get_featurizer(model_name)
	train_dataset = featurizer.featurize(train_dataset)
	train_dataset.get_shape()
	test_dataset = featurizer.featurize(test_dataset)
	test_dataset.get_shape()

	# Define type of prediction task
	if len(set(train_dataset.y)) > 2:
		mode = 'regression'
	else:
		mode = 'classification'

	# Get model build function
	if saved_model is None:
		print('Getting model build function')
		builder = BUILDERS[model_name]
		with open(hyperparams, 'rb') as f:
			hyperparams_dict = pickle.load(f)
		hyperparam_vals = hyperparams_dict[model_name]
		if model_name in ['GraphConv', 'TextCNN', 'Weave', 'MPNN', 'GCN', 'GAT', 'AttentiveFP', 'TorchMPNN']:
			model = builder(**hyperparam_vals)
		else:
			model = KerasModel(model_builder=builder, mode=mode, **hyperparam_vals)
		# Fit model
		print('Fitting model')
		model.fit(train_dataset)
		# model.save()
	else:
		model = load_from_disk(saved_model) # not sure this would work, as this fucntion loads models saved using joblib and I don't think that works for Keras models

	# Predict values
	predictions = model.predict(test_dataset)
	if mode == 'classification':
		predictions = normalize_labels_shape(predictions)
	df = pd.DataFrame(data={'mols': test_dataset.mols,
		                    'y_true': test_dataset.y,
	                        'y_pred': predictions})
	df.to_csv(os.path.join(output_dir, 'predictions.csv'))

	# Interpret model
	print('Calculating SHAP values')
	interpreter = ShapValues(dataset=test_dataset,
	                         model=model,
	                         mode=mode)
	interpreter.computeDeepShap(train_dataset=train_dataset,
	                            n_background_samples=100,
	                            plot=False)
	# interpreter.computeGradientShap()
	interpreter.plotFeatureExplanation(index='all', save=True, output_dir=output_dir, max_display=30)
	interpreter.plotBar(save=True, output_dir=output_dir, max_display=30)
	# interpreter.plotSampleExplanation(index=0, save=True, output_dir=output_dir) # need to find a good sample to explain
	interpreter.save_explanation_object(os.path.join(output_dir, 'shap_values_explainer.pkl'))
	interpreter.save_shap_values(os.path.join(output_dir, 'shap_values.csv'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train and evaluate user-defined models')
	parser.add_argument('-i',
	                    '--dataset-dir',
	                    type=str,
	                    help='Path to the directory where the train and test sets are stored for a particular dataset')
	parser.add_argument('-m',
	                    '--model-name',
	                    type=str,
	                    help='The type of model',
	                    default=None)
	parser.add_argument('-s',
	                    '--saved-model',
	                    type=str,
	                    help='Path to saved model',
	                    default=None)
	parser.add_argument('-p',
	                    '--hyperparams',
	                    type=str,
	                    help='Path to file containing the hyperparameter values to use')
	parser.add_argument('-g',
	                    '--gpu',
	                    type=str,
	                    help='The GPU to use')
	args = vars(parser.parse_args())
	main(**args)
