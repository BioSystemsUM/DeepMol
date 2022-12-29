'''author: Bruno Pereira
date: 28/04/2021
'''
import os
from unittest import TestCase

from deepchem.models import GraphConvModel
from deepmol.compound_featurization import MorganFingerprint
# from compound_featurization.rdkitFingerprints import RDKFingerprint, AtomPairFingerprint
from deepmol.compound_featurization import ConvMolFeat, WeaveFeat, SmileImageFeat, \
    MolGraphConvFeat
# from compound_featurization.mol2vec import Mol2Vec
# from datasets.datasets import NumpyDataset
from deepmol.loaders.loaders import CSVLoader
from deepmol.feature_selection import LowVarianceFS
from deepmol.splitters.splitters import SingletaskStratifiedSplitter
# from models.sklearnModels import SklearnModel
from deepmol.models.deepchem_models import DeepChemModel
from deepmol.metrics.metrics import Metric
from deepmol.metrics.metrics_functions import roc_auc_score, precision_score, accuracy_score
from deepmol.parameter_optimization.hyperparameter_optimization import HyperparameterOptimizerValidation, \
    HyperparameterOptimizerCV
# import preprocessing as preproc
from deepmol.utils import utils as preproc
# from imbalanced_learn.ImbalancedLearn import RandomOverSampler
# from deepchem.feat import WeaveFeaturizer, CoulombMatrix
# from deepchem.utils.conformers import ConformerGenerator
# from deepchem.trans import IRVTransformer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# from rdkit import Chem


# ds = MorganFingerprint().featurize(ds)
# ds = MACCSkeysFingerprint().featurize(ds)
# ds = LayeredFingerprint().featurize(ds)
# ds = RDKFingerprint().featurize(ds)
# ds = AtomPairFingerprint().featurize(ds)
# ds = Mol2Vec().featurize(ds)

print('-----------------------------------------------------')


# ds.get_shape()

# ds = LowVarianceFS(0.15).feature_selection(ds)

# ds = KbestFS().feature_selection(ds)
# ds = PercentilFS().feature_selection(ds)
# ds = RFECVFS().feature_selection(ds)
# ds = SelectFromModelFS().feature_selection(ds)

# train_dataset = RandomOverSampler().sample(train_dataset)


# k_folds = splitter.k_fold_split(ds, 3)

# for a, b in k_folds:
#    print(a.get_shape())
#    print(b.get_shape())
#    print('############')


# print(train_dataset.X)
# print(train_dataset.y)
# print(train_dataset.ids)
# print(train_dataset.features)
# print(train_dataset.features2keep)


# def rf_model_builder(n_estimators, max_features, class_weight, model_dir=None):
#     rf_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, class_weight=class_weight)
#     return SklearnModel(rf_model, model_dir)
#
# params_dict_rf = {"n_estimators": [10, 100],
#                   "max_features": ["auto", "sqrt", "log2", None],
#                   "class_weight": [{0: 1., 1: 1.}, {0: 1., 1: 5}, {0: 1., 1: 10}]
#                   }
#
# def svm_model_builder(C, gamma, kernel, model_dir=None):
#     svm_model = SVC(C=C, gamma=gamma, kernel=kernel)
#     return SklearnModel(svm_model, model_dir)
#
# params_dict_svm = {'C': [1.0, 0.7, 0.5, 0.3, 0.1],
#                'gamma': ["scale", "auto"],
#                'kernel': ["linear", "rbf"]
#               }
#
# optimizer = GridHyperparamOpt(rf_model_builder)
#
# best_rf, best_hyperparams, all_results = optimizer.hyperparam_search(params_dict_rf, train_dataset, valid_dataset, Metric(roc_auc_score))
#
# print('#################')
# print(best_hyperparams)
# print(best_rf)
#
# #print(best_rf.predict(test_dataset))
# print('@@@@@@@@@@@@@@@@')
# print(best_rf.evaluate(test_dataset, metrics))
#
# print(best_rf.predict(test_dataset))


def multitaskclass(dataset):
    from deepchem.models import MultitaskClassifier
    ds = MorganFingerprint().featurize(dataset)
    ds = LowVarianceFS(0.15).featureSelection(ds)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)
    multitask = MultitaskClassifier(n_tasks=1, n_features=np.shape(train_dataset.X)[1], layer_sizes=[1000])
    model_multi = DeepChemModel(multitask)
    # Model training
    model_multi.fit(train_dataset)
    valid_preds = model_multi.predict(valid_dataset)
    test_preds = model_multi.predict(test_dataset)
    # Evaluation
    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
    print('Training Dataset: ')
    train_score = model_multi.evaluate(train_dataset, metrics)
    print('Valid Dataset: ')
    valid_score = model_multi.evaluate(valid_dataset, metrics)
    print('Test Dataset: ')
    test_score = model_multi.evaluate(test_dataset, metrics)
    return


def graphconvmodel(dataset):
    from deepchem.models import GraphConvModel
    ds = ConvMolFeat().featurize(dataset)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)
    graph = GraphConvModel(n_tasks=1, mode='classification')
    model_graph = DeepChemModel(graph)
    # Model training
    model_graph.fit(train_dataset)
    valid_preds = model_graph.predict(valid_dataset)
    test_preds = model_graph.predict(test_dataset)
    # Evaluation
    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
    print('Training Dataset: ')
    train_score = model_graph.evaluate(train_dataset, metrics)
    print('Valid Dataset: ')
    valid_score = model_graph.evaluate(valid_dataset, metrics)
    print('Test Dataset: ')
    test_score = model_graph.evaluate(test_dataset, metrics)
    return


def mpnnmodel(dataset):
    from deepchem.models import MPNNModel
    ds = WeaveFeat().featurize(dataset)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)
    mpnn = MPNNModel(n_tasks=1, n_pair_feat=14, n_atom_feat=75, n_hidden=75, T=1, M=1, mode='classification')
    model_mpnn = DeepChemModel(mpnn)
    # Model training
    model_mpnn.fit(train_dataset)
    valid_preds = model_mpnn.predict(valid_dataset)
    test_preds = model_mpnn.predict(test_dataset)
    # Evaluation
    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
    print('Training Dataset: ')
    train_score = model_mpnn.evaluate(train_dataset, metrics)
    print('Valid Dataset: ')
    valid_score = model_mpnn.evaluate(valid_dataset, metrics)
    print('Test Dataset: ')
    test_score = model_mpnn.evaluate(test_dataset, metrics)
    return


def weavemodel(dataset):
    from deepchem.models import WeaveModel
    ds = WeaveFeat().featurize(dataset)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)
    weave = WeaveModel(n_tasks=1, mode='classification')
    model_weave = DeepChemModel(weave)
    # Model training
    model_weave.fit(train_dataset)
    valid_preds = model_weave.predict(valid_dataset)
    test_preds = model_weave.predict(test_dataset)
    # Evaluation
    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
    print('Training Dataset: ')
    train_score = model_weave.evaluate(train_dataset, metrics)
    print('Valid Dataset: ')
    valid_score = model_weave.evaluate(valid_dataset, metrics)
    print('Test Dataset: ')
    test_score = model_weave.evaluate(test_dataset, metrics)
    return


def chemcepmodel(dataset):
    ds = SmileImageFeat().featurize(dataset)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)
    chem = ChemCeptionModel(n_tasks=1, mode='classification')
    model_chem = DeepChemModel(chem)
    # Model training
    model_chem.fit(train_dataset)
    valid_preds = model_chem.predict(valid_dataset)
    test_preds = model_chem.predict(test_dataset)
    # Evaluation
    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
    print('Training Dataset: ')
    train_score = model_chem.evaluate(train_dataset, metrics)
    print('Valid Dataset: ')
    valid_score = model_chem.evaluate(valid_dataset, metrics)
    print('Test Dataset: ')
    test_score = model_chem.evaluate(test_dataset, metrics)
    return


def cnnmodel(dataset):
    ds = SmileImageFeat().featurize(dataset)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)
    cnn = CNNModel(n_tasks=1, n_features=np.shape(ds.X)[1], dims=1)
    model_cnn = DeepChemModel(cnn)
    # Model training
    model_cnn.fit(train_dataset)
    valid_preds = model_cnn.predict(valid_dataset)
    test_preds = model_cnn.predict(test_dataset)
    # Evaluation
    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
    print('Training Dataset: ')
    train_score = model_cnn.evaluate(train_dataset, metrics)
    print('Valid Dataset: ')
    valid_score = model_cnn.evaluate(valid_dataset, metrics)
    print('Test Dataset: ')
    test_score = model_cnn.evaluate(test_dataset, metrics)
    return


def smilesvec(dataset):
    from deepchem.models import Smiles2Vec
    ds = SmileSeqFeat().featurize(dataset)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)
    vec = Smiles2Vec(ds.dictionary, n_tasks=1, mode='classification')
    model_vec = DeepChemModel(vec)
    # Model training
    model_vec.fit(train_dataset)
    valid_preds = model_vec.predict(valid_dataset)
    test_preds = model_vec.predict(test_dataset)
    # Evaluation
    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
    print('Training Dataset: ')
    train_score = model_vec.evaluate(train_dataset, metrics)
    print('Valid Dataset: ')
    valid_score = model_vec.evaluate(valid_dataset, metrics)
    print('Test Dataset: ')
    test_score = model_vec.evaluate(test_dataset, metrics)
    return


def irvmodel(dataset):
    from deepchem.models import MultitaskIRVClassifier
    ds = MorganFingerprint().featurize(dataset)
    ds = preproc.irv_transformation(ds, K=10, n_tasks=1)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)
    irv = MultitaskIRVClassifier(n_tasks=1, mode='classification')
    model_irv = DeepChemModel(irv)
    # Model training
    model_irv.fit(train_dataset)
    valid_preds = model_irv.predict(valid_dataset)
    test_preds = model_irv.predict(test_dataset)
    # Evaluation
    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
    print('Training Dataset: ')
    train_score = model_irv.evaluate(train_dataset, metrics)
    print('Valid Dataset: ')
    valid_score = model_irv.evaluate(valid_dataset, metrics)
    print('Test Dataset: ')
    test_score = model_irv.evaluate(test_dataset, metrics)
    return


def gatmodel(dataset):
    from deepchem.models import GATModel
    ds = MolGraphConvFeat().featurize(dataset)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)
    gat = GATModel(n_tasks=1, mode='classification')
    model_gat = DeepChemModel(gat)
    # Model training
    model_gat.fit(train_dataset)
    valid_preds = model_gat.predict(valid_dataset)
    test_preds = model_gat.predict(test_dataset)
    # Evaluation
    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
    print('Training Dataset: ')
    train_score = model_gat.evaluate(train_dataset, metrics)
    print('Valid Dataset: ')
    valid_score = model_gat.evaluate(valid_dataset, metrics)
    print('Test Dataset: ')
    test_score = model_gat.evaluate(test_dataset, metrics)
    return


def gcnmodel(dataset):
    ds = MolGraphConvFeat().featurize(dataset)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)
    gcn = CNModel(n_tasks=1, mode='classification')
    model_gcn = DeepChemModel(gcn)
    # Model training
    model_gcn.fit(train_dataset)
    valid_preds = model_gcn.predict(valid_dataset)
    test_preds = model_gcn.predict(test_dataset)
    # Evaluation
    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
    print('Training Dataset: ')
    train_score = model_gcn.evaluate(train_dataset, metrics)
    print('Valid Dataset: ')
    valid_score = model_gcn.evaluate(valid_dataset, metrics)
    print('Test Dataset: ')
    test_score = model_gcn.evaluate(test_dataset, metrics)
    return


def attmodel(dataset):
    from deepchem.models import AttentiveFPModel
    ds = MolGraphConvFeat(use_edges=True).featurize(dataset)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)
    att = AttentiveFPModel(n_tasks=1, mode='classification')
    model_att = DeepChemModel(att)
    # Model training
    model_att.fit(train_dataset)
    valid_preds = model_att.predict(valid_dataset)
    test_preds = model_att.predict(test_dataset)
    # Evaluation
    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
    print('Training Dataset: ')
    train_score = model_att.evaluate(train_dataset, metrics)
    print('Valid Dataset: ')
    valid_score = model_att.evaluate(valid_dataset, metrics)
    print('Test Dataset: ')
    test_score = model_att.evaluate(test_dataset, metrics)
    return


def dagmodel(dataset):
    from deepchem.models import DAGModel
    ds = ConvMolFeat().featurize(dataset)
    ds = preproc.dag_transformation(ds, max_atoms=150)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)
    dag = DAGModel(n_tasks=1, max_atoms=150, mode='classification')
    model_dag = DeepChemModel(dag)
    # Model training
    model_dag.fit(train_dataset)
    valid_preds = model_dag.predict(valid_dataset)
    test_preds = model_dag.predict(test_dataset)
    # Evaluation
    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
    print('Training Dataset: ')
    train_score = model_dag.evaluate(train_dataset, metrics)
    print('Valid Dataset: ')
    valid_score = model_dag.evaluate(valid_dataset, metrics)
    print('Test Dataset: ')
    test_score = model_dag.evaluate(test_dataset, metrics)
    return


def graphconvbuilder(graph_conv_layers, dense_layer_size, dropout, model_dir=None):
    from deepchem.models import GraphConvModel
    graph = GraphConvModel(n_tasks=1,
                           graph_conv_layers=graph_conv_layers,
                           dense_layer_size=dense_layer_size,
                           dropout=dropout)
    return DeepChemModel(graph)


def hyperoptimgraph(dataset):
    ds = ConvMolFeat().featurize(dataset)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)

    params = {'graph_conv_layers': [[64, 64], [72, 72], [84, 84]],
              'dense_layer_size': [128, 144, 198],
              'dropout': [0.0, 0.25, 0.5]}

    optimizer = HyperparameterOptimizerValidation(graphconvbuilder)

    best_rf, best_hyperparams, all_results = optimizer.hyperparameter_search(params, train_dataset, valid_dataset,
                                                                             Metric(roc_auc_score))

    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]

    print('Best Model: ')
    print(best_rf.evaluate(test_dataset, metrics))
    return


def mpnnbuilder(n_atom_feat, n_pair_feat, n_hidden, T, M, dropout, model_dir=None):
    from deepchem.models import MPNNModel
    mpnn = MPNNModel(n_tasks=1,
                     n_atom_feat=n_atom_feat,
                     n_pair_feat=n_pair_feat,
                     n_hidden=n_hidden,
                     T=T,
                     M=M,
                     dropout=dropout,
                     mode='classification')
    return DeepChemModel(mpnn)


def hyperoptimmpnn(dataset):
    ds = WeaveFeat().featurize(dataset)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)

    params = {'n_atom_feat': [45],
              'n_pair_feat': [14],
              'n_hidden': [50, 75, 100],
              'T': [1, 10],
              'M': [1, 10],
              'dropout': [0.0, 0.25, 0.5]}

    optimizer = HyperparamOpt_Val(mpnnbuilder)

    best_rf, best_hyperparams, all_results = optimizer.hyperparameter_search(params, train_dataset, valid_dataset,
                                                                             Metric(roc_auc_score))

    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]

    print('Best Model: ')
    print(best_rf.evaluate(test_dataset, metrics))
    return


def gatbuilder(n_attention_heads, dropout, alpha, predictor_hidden_feats, predictor_dropout, number_atom_features,
               model_dir=None):
    from deepchem.models import GATModel
    gat = GATModel(n_tasks=1,
                   n_attention_heads=n_attention_heads,
                   dropout=dropout,
                   alpha=alpha,
                   predictor_hidden_feats=predictor_hidden_feats,
                   predictor_dropout=predictor_dropout,
                   number_atom_features=number_atom_features,
                   mode='classification')
    return DeepChemModel(gat)


def hyperoptimgat(dataset):
    ds = MolGraphConvFeat().featurize(dataset)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)

    params = {'n_attention_heads': [8, 16],
              'dropout': [0.0, 0.25, 0.5],
              'alpha': [0.2, 0.4],
              'predictor_hidden_feats': [128, 256],
              'predictor_dropout': [0.0, 0.25],
              'number_atom_features': [30, 45]}
    optimizer = HyperparameterOptimizerValidation(gatbuilder)

    best_rf, best_hyperparams, all_results = optimizer.hyperparameter_search(params, train_dataset, valid_dataset,
                                                                             Metric(roc_auc_score))

    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]

    print('Best Model: ')
    print(best_rf.evaluate(test_dataset, metrics))
    return


def gcnbuilder(graph_conv_layers, dropout, predictor_hidden_feats, predictor_dropout, number_atom_features,
               learning_rate, model_dir=None):
    from deepchem.models import GCNModel
    gcn = GCNModel(n_tasks=1,
                   graph_conv_layers=graph_conv_layers,
                   dropout=dropout,
                   predictor_hidden_feats=predictor_hidden_feats,
                   predictor_dropout=predictor_dropout,
                   number_atom_features=number_atom_features,
                   learning_rate=learning_rate,
                   mode='classification')
    return DeepChemModel(gcn)


def hyperoptimgcn(dataset):
    ds = MolGraphConvFeat().featurize(dataset)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)

    params = {'graph_conv_layers': [[64, 64], [72, 72], [84, 84]],
              'dropout': [0.0, 0.25, 0.50],
              'predictor_hidden_feat': [128, 256],
              'predictor_dropout': [0.0, 0.25],
              'number_atom_features': [30, 45],
              'learning_rate': [0.001, 0.01]}
    optimizer = HyperparameterOptimizerValidation(gcnbuilder)

    best_rf, best_hyperparams, all_results = optimizer.hyperparameter_search(params, train_dataset, valid_dataset,
                                                                             Metric(roc_auc_score))

    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]

    print('Best Model: ')
    print(best_rf.evaluate(test_dataset, metrics))
    return


def cnnbuilder(n_features, layer_filters, kernel_size, weight_init_stddevs, bias_init_consts, weight_decay_penalty,
               dropouts, model_dir=None):
    cnn = CNNModel(n_tasks=1,
                   n_features=n_features,
                   layer_filters=layer_filters,
                   kernel_size=kernel_size,
                   weight_init_stddevs=weight_init_stddevs,
                   bias_init_consts=bias_init_consts,
                   weight_decay_penalty=weight_decay_penalty,
                   dropouts=dropouts)
    return DeepChemModel(cnn)


def hyperoptimcnn(dataset):
    ds = SmileImageFeat().featurize(dataset)
    splitter = SingletaskStratifiedSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)

    params = {'n_features': np.shape(ds.X)[1],
              'layer_filters': [[100], [150], [200]],
              'kernel_size': [5, 10],
              'weight_init_stddevs': [0.02, 0.04],
              'bias_init_consts': [1.0, 2.0],
              'weight_decay_penalty': [0.0, 0.25],
              'dropouts': [0.25, 0.5, 0.75]}
    optimizer = HyperparameterOptimizerValidation(cnnbuilder)

    best_rf, best_hyperparams, all_results = optimizer.hyperparameter_search(params, train_dataset, valid_dataset,
                                                                             Metric(roc_auc_score))

    metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]

    print('Best Model: ')
    print(best_rf.evaluate(test_dataset, metrics))
    return


def menu():
    ds = None
    string = '''
    This is a file that allows to test multiple DeepChem Models.
    1 - Read the dataset (only performed once)
    2 - Print shape of dataset
    3 - Select a Model (featurization and splitting included)
    4 - Hyperparameter optimization
    5 - Exit
    '''
    substring = '''
    Models available:
    a - MultitaskClassifier
    b - GraphConvModel
    c - MPNNModel
    d - WeaveModel
    e - ChemCeption
    f - CNN
    g - Smiles2Vec
    h - MultitaskIRVClassifier
    i - GATModel
    j - GCNModel 
    k - AttentiveFPModel
    l - DAGModel
    m - Return
    '''
    substring2 = '''
    Models available:
    a - GraphConvModel
    b - MPNNModel
    c - GATModel
    d - GCNModel
    e - CNN
    '''
    while True:
        print(string)
        opt = int(input('Option: '))
        if opt == 1:
            if ds is None:
                dataset = CSVLoader(dataset_path='tests/data/preprocessed_dataset_wfoodb.csv',
                                    mols_field='Smiles',
                                    labels_fields='Class',
                                    id_field='ID')  # , shard_size=4000)
                ds = dataset.create_dataset()
                print('Dataset established')
            else:
                print('Dataset already read')
        elif opt == 2:
            if ds is None:
                print('A dataset has to be read first')
            else:
                ds.get_shape()
                # print('X: ', X)
                # print('y: ', y)
                # print('features: ', features)
                # print('ids: ', ids)
        elif opt == 3 and ds is not None:
            print(substring)
            opt2 = input('Model (letter): ')
            if opt2 == 'a':
                multitaskclass(ds)
            elif opt2 == 'b':
                graphconvmodel(ds)
            elif opt2 == 'c':
                mpnnmodel(ds)
            elif opt2 == 'd':
                weavemodel(ds)
            elif opt2 == 'e':
                chemcepmodel(ds)
            elif opt2 == 'f':
                cnnmodel(ds)
            elif opt2 == 'g':
                smilesvec(ds)
            elif opt2 == 'h':
                irvmodel(ds)
            elif opt2 == 'i':
                gatmodel(ds)
            elif opt2 == 'j':
                gcnmodel(ds)
            elif opt2 == 'k':
                attmodel(ds)
            elif opt2 == 'l':
                dagmodel(ds)
            elif opt2 == 'm':
                pass
            else:
                print('Invalid option')
        elif opt == 4:
            if ds is None:
                print('A dataset has to be read first')
            else:
                print(substring2)
                opt3 = input('Model (letter): ')
                if opt3 == 'a':
                    hyperoptimgraph(ds)
                elif opt3 == 'b':
                    hyperoptimmpnn(ds)
                elif opt3 == 'c':
                    hyperoptimgat(ds)
                elif opt3 == 'd':
                    hyperoptimgcn(ds)
                elif opt3 == 'e':
                    hyperoptimcnn(ds)
        elif opt == 5:
            break


if __name__ == '__main__':
    menu()
