import os

import pandas as pd

from src.compoundFeaturization import rdkitFingerprints, mol2vec, deepChemFeaturizers


def save_evaluation_results(test_results_dict, train_results_dict, hyperparams, model_name, model_dir, output_filepath):
    results = {'model': model_name, 'model_dir': model_dir}
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
                        'LayeredFP': rdkitFingerprints.LayeredFingerprint(fpSize=1024)}
    return featurizers_dict[model_name]


def get_default_param_grid(model_name):
    grids_dict = {'GraphConv': {'graph_conv_layers': [[32, 32], [64, 64], [128, 128],
                                                      [32, 32, 32], [64, 64, 64], [128, 128, 128],
                                                      [32, 32, 32, 32], [64, 64, 64, 64], [128, 128, 128, 128]],
                                'dense_layer_size': [2048, 1024, 512, 256, 128, 64, 32],
                                'dropout': [0.0, 0.25, 0.5],
                                'learning_rate': [1e-4, 1e-3, 1e-2]},
                  'TextCNN': {'n_embedding': [75, 32, 64],
                              'kernel_sizes': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],  # DeepChem default. Their code says " Multiple convolutional layers with different filter widths", so I'm not repeating kernel_sizes
                                               [1, 2, 3, 4, 5, 7, 10, 15],
                                               [3, 4, 5, 7, 10, 15],
                                               [3, 4, 5, 7, 10],
                                               [3, 4, 5, 7],
                                               [3, 4, 5],
                                               [3, 5, 7]],
                              'num_filters': [[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160], # DeepChem default
                                              [32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128],
                                              [128, 128, 128, 128, 64, 64, 64, 64, 32, 32, 32, 32]],
                              'dropout': [0.0, 0.25, 0.5],
                              'learning_rate': [1e-4, 1e-3, 1e-2]},
                  'Weave': {'n_hidden': [32, 64, 128],
                            'n_graph_feat': [128, 256],
                            'n_weave': [2, 3, 4],
                            'fully_connected_layer_sizes': [[2048], [1024], [512], [256]],
                            'dropouts': [0.0, 0.25, 0.5],
                            'learning_rate': [1e-4, 1e-3, 1e-2]},
                  'ECFP4': {'hlayers_sizes': ['[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
                                              '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
                                              '[512, 256, 128, 64]', '[256, 128, 64, 32]'],
                            'l2': [0.0, 1e-4, 1e-3, 1e-2],
                            'hidden_dropout': [0.0, 0.25, 0.5],
                            'batchnorm': [True, False],
                            'learning_rate': [1e-4, 1e-3, 1e-2]},
                  'ECFP6': {'hlayers_sizes': ['[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
                                              '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
                                              '[512, 256, 128, 64]', '[256, 128, 64, 32]'],
                            'l2': [0.0, 1e-4, 1e-3, 1e-2],
                            'hidden_dropout': [0.0, 0.25, 0.5],
                            'batchnorm': [True, False],
                            'learning_rate': [1e-4, 1e-3, 1e-2]},
                  'Mol2vec': {'hlayers_sizes': ['[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
                                                '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]',
                                                '[256, 128, 64, 32]', '[128, 64, 32, 16]'],
                              'l2': [0.0, 1e-4, 1e-3, 1e-2],
                              'hidden_dropout': [0.0, 0.25, 0.5],
                              'batchnorm': [True, False],
                              'learning_rate': [1e-4, 1e-3, 1e-2]},
                  'MACCS': {'hlayers_sizes': ['[128, 64]', '[64, 32]', '[32, 16]', '[16, 8]',
                                              '[128, 64, 32]', '[64, 32, 16]', '[32, 16, 8]',
                                              '[128, 64, 32, 16]', '[64, 32, 16, 8]'],
                            'l2': [0.0, 1e-4, 1e-3, 1e-2],
                            'hidden_dropout': [0.0, 0.25, 0.5],
                            'batchnorm': [True, False],
                            'learning_rate': [1e-4, 1e-3, 1e-2]},
                  'RDKitFP': {'hlayers_sizes': ['[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
                                              '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
                                              '[512, 256, 128, 64]', '[256, 128, 64, 32]'],
                              'l2': [0.0, 1e-4, 1e-3, 1e-2],
                              'hidden_dropout': [0.0, 0.25, 0.5],
                              'batchnorm': [True, False],
                              'learning_rate': [1e-4, 1e-3, 1e-2]},
                  'MPNN': {'n_hidden': [128, 256, 100], # 100 is DeepChem's default value
                           'T': [3, 4, 5, 6, 7, 8], # as mentioned in the Gilmer et al, 2017 paper
                           'M': [1, 2, 3, 4, 5, 6, 7, 8, 9, 12], # as mentioned in the Gilmer et al, 2017 paper
                           'learning_rate': [1e-5, 1e-4, 1e-3]}, # because Gilmer et al, 2017 paper mentions lower values
                  'GCN': {'graph_conv_layers': [[32, 32], [64, 64], [128, 128],
                                                [32, 32, 32], [64, 64, 64], [128, 128, 128],
                                                [32, 32, 32, 32], [64, 64, 64, 64], [128, 128, 128, 128]],
                          'predictor_hidden_feats': [256, 128, 64],
                          'dropout': [0.0, 0.25, 0.5],
                          'predictor_dropout': [0.0, 0.25, 0.5],
                          'learning_rate': [1e-4, 1e-3, 1e-2]},
                  'GAT': {'graph_attention_layers': [[8, 8], [16, 16], [32, 32],
                                                     [8, 8, 8], [16, 16, 16], [32, 32, 32]],
                          'n_attention_heads': [4, 8],
                          'predictor_hidden_feats': [256, 128, 64],
                          'dropout': [0.0, 0.25, 0.5],
                          'predictor_dropout': [0.0, 0.25, 0.5],
                          'learning_rate': [1e-4, 1e-3, 1e-2]},
                  'AttentiveFP': {'num_layers': [1, 2, 3, 4],
                                  'num_timesteps': [2, 3, 4],
                                  'graph_feat_size': [32, 64, 128, 256, 512, 200], # 200 is the default value in DeepChem
                                  'dropout': [0, 0.25, 0.5],
                                  'learning_rate': [1e-4, 1e-3, 1e-2]},
                  'TorchMPNN': {'node_out_feats': [64, 128],
                                'edge_hidden_feats': [128, 256],
                                'num_step_message_passing': [3, 4, 5, 6, 7, 8],
                                'num_step_set2set': [2, 4, 6, 8],
                                'num_layer_set2set': [2, 3],
                                'learning_rate': [1e-5, 1e-4, 1e-3]},
                  'AtomPair': {'hlayers_sizes': ['[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
                                              '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
                                              '[512, 256, 128, 64]', '[256, 128, 64, 32]'],
                              'l2': [0.0, 1e-4, 1e-3, 1e-2],
                              'hidden_dropout': [0.0, 0.25, 0.5],
                              'batchnorm': [True, False],
                              'learning_rate': [1e-4, 1e-3, 1e-2]},
                  'LayeredFP': {'hlayers_sizes': ['[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
                                              '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
                                              '[512, 256, 128, 64]', '[256, 128, 64, 32]'],
                              'l2': [0.0, 1e-4, 1e-3, 1e-2],
                              'hidden_dropout': [0.0, 0.25, 0.5],
                              'batchnorm': [True, False],
                              'learning_rate': [1e-4, 1e-3, 1e-2]}
                  }
    return grids_dict[model_name]