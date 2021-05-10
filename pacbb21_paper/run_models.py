import argparse
import os
from datetime import datetime

import pandas as pd
import tensorflow as tf
from deepchem.models import TextCNNModel

from src.loaders.Loaders import CSVLoader
from src.metrics.Metrics import Metric
from src.metrics.metricsFunctions import r2_score, mean_squared_error, pearson_score, accuracy_score, roc_auc_score, \
    prc_auc_score, precision_score, recall_score, spearman_score
from src.parameterOptimization.HyperparameterOpt import HyperparamOpt_CV

from model_build_functions import BUILDERS
from utils import get_featurizer, save_evaluation_results, get_default_param_grid


def main(dataset_dir, model_name, gpu):
    print(tf.__version__)
    print(tf.test.is_built_with_cuda())
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_visible_devices(gpus[int(gpu)], 'GPU')
    #     except:
    #         print('Invalid device or cannot modify virtual devices once initialized.')
    # print(tf.config.get_visible_devices())

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # print(tf.reduce_sum(tf.random.normal([1000, 1000])))

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

    full_dataset = train_dataset.merge([test_dataset])  # only necessary for TextCNN models (to build char_dict)

    # Featurization
    print('Featurizing molecules')
    featurizer = get_featurizer(model_name)
    train_dataset = featurizer.featurize(train_dataset)
    train_dataset.get_shape()
    test_dataset = featurizer.featurize(test_dataset)
    test_dataset.get_shape()

    full_dataset = featurizer.featurize(full_dataset)  # only necessary for TextCNN models (to build char_dict)

    # Define type of prediction task and define scoring metrics accordingly
    if len(set(train_dataset.y)) > 2:
        mode = 'regression'
        metrics = [Metric(mean_squared_error, n_tasks=1),
                   Metric(r2_score, n_tasks=1),
                   Metric(pearson_score, mode='regression', n_tasks=1),
                   Metric(spearman_score, mode='regression', n_tasks=1)]
        opt_metric = 'neg_mean_squared_error'
    else:
        mode = 'classification'
        metrics = [Metric(roc_auc_score, n_tasks=1),
                   Metric(prc_auc_score, n_tasks=1),
                   Metric(accuracy_score, n_tasks=1),
                   Metric(precision_score, n_tasks=1),
                   Metric(recall_score, n_tasks=1)]
        opt_metric = 'roc_auc'

    # Get model build function
    builder = BUILDERS[model_name]

    # Hyperparameter Optimization with CV
    print('Optimizing hyperparameters')
    params_dict = get_default_param_grid(model_name)
    # Add some necessary hyperparameters to params_dict
    params_dict['task_type'] = [mode] # calling it "task_type" so that it doesn't conflict with KerasModel's __init__
    params_dict['epochs'] = [100]
    params_dict['batch_size'] = [256]
    if builder.__name__ == 'dense_builder':
        params_dict['input_dim'] = [train_dataset.len_X()[1]]
    if model_name == 'TextCNN':
        char_dict, length = TextCNNModel.build_char_dict(full_dataset)
        params_dict['char_dict'] = [char_dict]
        params_dict['seq_length'] = [length]

    optimizer = HyperparamOpt_CV(builder, mode=mode)

    if model_name in ['GraphConv', 'TextCNN', 'Weave', 'MPNN']:
        model_type = 'deepchem'
    else:
        model_type = 'keras'

    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(model_type,
                                                                            params_dict,
                                                                            train_dataset,
                                                                            opt_metric,
                                                                            cv=5,
                                                                            n_iter_search=30,
                                                                            n_jobs=1,
                                                                            seed=123) # so that the folds are always the same for all models

    print('#################')
    print(best_hyperparams)
    print(best_model)
    grid_results_df = pd.DataFrame(data=all_results)
    grid_results_df.to_csv(os.path.join(output_dir, 'hyperparam_opt_results.csv'))

    # Evaluate model
    # (best_model has already been fit)
    print('Evaluating model on test set')
    train_scores = best_model.evaluate(train_dataset, metrics)
    test_scores = best_model.evaluate(test_dataset, metrics)
    save_evaluation_results(test_scores, train_scores, best_hyperparams, model_name, model_dir=output_dir,
                            output_filepath='../pacbb21_paper/results/%s_evaluation_results.csv' % dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate user-defined models')
    parser.add_argument('-i',
                        '--dataset-dir',
                        type=str,
                        help='Path to the directory where the train and test sets are stored for a particular dataset')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        help='The type of model')
    parser.add_argument('-g',
                        '--gpu',
                        type=str,
                        help='The GPU to use')
    args = vars(parser.parse_args())
    main(**args)