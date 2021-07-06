import os
from ast import literal_eval
from copy import copy
import pandas as pd

import yaml
from deepchem.metrics import prc_auc_score
from deepchem.models import TextCNNModel
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, mean_squared_error, r2_score
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.regularizers import l1_l2
from yaml.loader import SafeLoader

# Open the file and load the file
from compoundFeaturization import rdkitFingerprints, deepChemFeaturizers, mol2vec
from compoundFeaturization.mixedDescriptors import MixedFeaturizer
from compoundFeaturization.mordredDescriptors import Mordred3DFeaturizer, MordredFeaturizer
from compoundFeaturization.rdkit3DDescriptors import All3DDescriptors
from compoundFeaturization.rdkitFingerprints import LayeredFingerprint
from loaders.Loaders import SDFLoader
from metrics.Metrics import Metric
from metrics.metricsFunctions import pearson_score, spearman_score
from parameterOptimization.HyperparameterOpt import HyperparamOpt_CV


def get_featurizer(model_name):
    featurizers_dict = {'GraphConv': deepChemFeaturizers.ConvMolFeat(),
                        'Weave': deepChemFeaturizers.WeaveFeat(),
                        'TextCNN': deepChemFeaturizers.RawFeat(),
                        'ECFP4': rdkitFingerprints.MorganFingerprint(radius=2, size=1024),
                        'ECFP6': rdkitFingerprints.MorganFingerprint(radius=3, size=1024),
                        'Mol2vec': mol2vec.Mol2Vec(pretrain_model_path='model_300dim.pkl'),
                        'MACCS': rdkitFingerprints.MACCSkeysFingerprint(),
                        'RDKitFP': rdkitFingerprints.RDKFingerprint(fpSize=1024),
                        'MPNN': deepChemFeaturizers.WeaveFeat(),
                        'GCN': deepChemFeaturizers.MolGraphConvFeat(),
                        'GAT': deepChemFeaturizers.MolGraphConvFeat(),
                        'AttentiveFP': deepChemFeaturizers.MolGraphConvFeat(use_edges=True),
                        'TorchMPNN': deepChemFeaturizers.MolGraphConvFeat(use_edges=True),
                        'AtomPair': rdkitFingerprints.AtomPairFingerprint(nBits=1024),
                        'LayeredFP': rdkitFingerprints.LayeredFingerprint(fpSize=1024),
                        'All3DDescriptors': All3DDescriptors()}

    return featurizers_dict[model_name]


def read_and_featurize_datasets(train_dataset_path, test_dataset_path, featurizers):
    loader = SDFLoader("data/" + train_dataset_path, labels_fields='_y')
    train_dataset = loader.create_dataset()

    loader = SDFLoader("data/" + test_dataset_path, labels_fields='_y')
    test_dataset = loader.create_dataset()

    class_featurizers = []
    scale = False
    for featurizer in featurizers:
        class_featurizers.append(get_featurizer(featurizer))
        if featurizer == "All3DDescriptors":
            scale = True

    if len(class_featurizers) > 1:
        train_dataset = MixedFeaturizer(class_featurizers).featurize(train_dataset, scale=scale)
        test_dataset = MixedFeaturizer(class_featurizers).featurize(test_dataset, scale=scale)

    elif "All3DDescriptors" not in class_featurizers:
        train_dataset = class_featurizers[0].featurize(train_dataset)
        test_dataset = class_featurizers[0].featurize(test_dataset)

    else:
        train_dataset = class_featurizers[0].featurize(train_dataset, scale=scale)
        test_dataset = class_featurizers[0].featurize(test_dataset, scale=scale)

    return train_dataset, test_dataset


def dense_builder(input_dim=None, task_type=None, hlayers_sizes='[10]', initializer='he_normal',
                  l1=0, l2=0, hidden_dropout=0, batchnorm=True, learning_rate=0.001):
    hlayers_sizes = literal_eval(hlayers_sizes)
    l2 = literal_eval(str(l2))
    learning_rate = literal_eval(str(learning_rate))
    # hlayers_sizes was passed as a str because Pipeline throws RuntimeError when cloning the model if parameters are lists

    model = Sequential()
    model.add(Input(shape=input_dim))

    for i in range(len(hlayers_sizes)):
        model.add(Dense(units=hlayers_sizes[i], kernel_initializer=initializer,
                        kernel_regularizer=l1_l2(l1=l1, l2=l2)))
        if batchnorm:
            model.add(BatchNormalization())

        model.add(Activation('relu'))

        if hidden_dropout > 0:
            model.add(Dropout(rate=hidden_dropout))

    # Add output layer
    if task_type == 'regression':
        model.add(Dense(1, activation='linear', kernel_initializer=initializer))
    elif task_type == 'classification':
        model.add(Dense(1, activation='sigmoid', kernel_initializer=initializer))

    # Define optimizer
    opt = Adam(lr=learning_rate)

    # Compile model
    if task_type == 'regression':
        model.compile(loss='mean_squared_error', optimizer=opt)
    else:
        model.compile(loss='binary_crossentropy', optimizer=opt)

    return model


def train_models_and_evaluate(train_dataset, test_dataset, params_dict, mode, model_name, dataset_name):
    from model_build_functions import BUILDERS

    if str(model_name[0]) not in BUILDERS.keys() or len(model_name) > 1:
        builder = dense_builder
    else:
        builder = BUILDERS[model_name[0]]

    full_dataset = train_dataset.merge([test_dataset])

    params_dict['task_type'] = [mode]  # calling it "task_type" so that it doesn't conflict with KerasModel's __init__
    params_dict['epochs'] = [10]
    params_dict['batch_size'] = [256]

    if builder.__name__ == 'dense_builder':
        params_dict['input_dim'] = [train_dataset.len_X()[1]]
    if model_name[0] == 'TextCNN':
        char_dict, length = TextCNNModel.build_char_dict(full_dataset)
        params_dict['char_dict'] = [char_dict]
        params_dict['seq_length'] = [length]
        # train_dataset.redefine_ids()
        # test_dataset.redefine_ids()

    if model_name[0] in ['GraphConv', 'TextCNN', 'Weave', 'MPNN', 'GCN', 'GAT', 'AttentiveFP', 'TorchMPNN']:
        model_type = 'deepchem'
    else:
        model_type = 'keras'
    print(model_type)

    optimizer = HyperparamOpt_CV(builder, mode=mode)

    if mode == "classification":

        opt_metric = 'roc_auc'

        metrics = [Metric(roc_auc_score, n_tasks=1),
                   Metric(accuracy_score, n_tasks=1),
                   Metric(precision_score, n_tasks=1),
                   Metric(recall_score, n_tasks=1)]
    else:

        metrics = [Metric(mean_squared_error, n_tasks=1),
                   Metric(r2_score, n_tasks=1),
                   Metric(pearson_score, mode='regression', n_tasks=1),
                   Metric(spearman_score, mode='regression', n_tasks=1)]

        opt_metric = 'neg_mean_squared_error'

    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(model_type,
                                                                            params_dict,
                                                                            train_dataset,
                                                                            opt_metric,
                                                                            cv=2,
                                                                            n_iter_search=1,
                                                                            n_jobs=1,
                                                                            seed=123, verbose=0)

    print('#################')
    print(best_hyperparams)

    output_filepath = "results_%s.csv" % dataset_name.replace(".csv", "")

    grid_results_df = pd.DataFrame(data=all_results)

    if os.path.exists('hyperparam_opt_results.csv'):
        saved_df = pd.read_csv('hyperparam_opt_results.csv')
        grid_results_df = pd.concat([saved_df, grid_results_df], axis=0, ignore_index=True, sort=False)

    grid_results_df.to_csv(os.path.join("", 'hyperparam_opt_results.csv'))

    # Evaluate model
    # (best_model has already been fit)
    print('Evaluating model on test set')
    train_scores = best_model.evaluate(train_dataset, metrics)
    test_scores = best_model.evaluate(test_dataset, metrics)

    output_dir = ""
    if output_filepath is None:
        output_filepath = os.path.join('3d_experiment', 'results', '%s_evaluation_results.csv' % dataset_name)
    save_evaluation_results(test_scores, train_scores, best_hyperparams, model_name, model_dir=output_dir,
                            output_filepath=output_filepath)


def save_evaluation_results(test_results_dict, train_results_dict, hyperparams, model_name, model_dir, output_filepath):
    results = {'model': "_".join(model_name), 'model_dir': model_dir}
    test_results_dict = {'test_' + k: [v] for k, v in test_results_dict.items()}
    train_results_dict = {'train_' + k: [v] for k, v in train_results_dict.items()}
    results.update(test_results_dict)
    results.update(train_results_dict)
    results.update({k: [v] for k, v in hyperparams.items()})
    results_df = pd.DataFrame(data=results)

    if os.path.exists(output_filepath):
        saved_df = pd.read_csv(output_filepath)
        results_df = pd.concat([saved_df, results_df], axis=0, ignore_index=True, sort=False)

    # change column order
    cols = results_df.columns.tolist()
    test_score_cols = sorted(test_results_dict.keys())
    train_score_cols = sorted(train_results_dict.keys())
    hyperparam_cols = sorted([x for x in cols if x not in ['model', 'model_dir'] + train_score_cols + test_score_cols])
    rearranged_cols = ['model', 'model_dir'] + test_score_cols + train_score_cols + hyperparam_cols
    results_df = results_df[rearranged_cols]
    results_df.to_csv(output_filepath, index=False)


def perform_experiment(train_dataset_path, test_dataset_path, featurization, hyperparameters, mode):
    # try:
    train_dataset, test_dataset = read_and_featurize_datasets(train_dataset_path,
                                                              test_dataset_path,
                                                              featurization)

    train_models_and_evaluate(
        train_dataset, test_dataset, hyperparameters,
        mode, featurization,
        train_dataset_path
    )
    # except:
    #     with open("datasets_experiment/errors", "a+") as f:
    #         f.write(
    #             featurization + " with " + datasets[mode][dataset]["train"][0] + "\n")


import tensorflow as tf

print(tf.__version__)
print(tf.test.is_built_with_cuda())
# print(torch.cuda.is_available())
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_visible_devices(gpus[int(gpu)], 'GPU')
#     except:
#         print('Invalid device or cannot modify virtual devices once initialized.')
# print(tf.config.get_visible_devices())


with open('pipeline.yml') as f:
    data = yaml.load(f, Loader=SafeLoader)

    datasets = data['datasets']

    preprocessing_and_hyperparameters = data['preprocessing_and_hyperparameters']

# os.environ['CUDA_VISIBLE_DEVICES'] = gpu
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

for mode in datasets:
    # mode = "regression"
    for dataset in datasets[mode]:
        for featurization_type in preprocessing_and_hyperparameters:

            if featurization_type == "deepchem":
                featurization_type_dict = preprocessing_and_hyperparameters[featurization_type]

                for elem in featurization_type_dict:
                    experiment = featurization_type_dict[elem]

                    featurization_experiment = experiment["featurization"]

                    if "TextCNN" not in featurization_experiment:

                        hyperparameters = experiment["hyperparameters"]

                        train_dataset_path = datasets[mode][dataset]["train"][0]
                        test_dataset_path = datasets[mode][dataset]["test"][0]

                        if isinstance(featurization_experiment, list):
                            perform_experiment(train_dataset_path, test_dataset_path,
                                               featurization_experiment, hyperparameters,
                                               mode)

                        else:
                            for feature in featurization_experiment:

                                featurization_experiment_elem = featurization_experiment[feature]

                                perform_experiment(train_dataset_path, test_dataset_path,
                                                   featurization_experiment_elem, hyperparameters,
                                                   mode)
