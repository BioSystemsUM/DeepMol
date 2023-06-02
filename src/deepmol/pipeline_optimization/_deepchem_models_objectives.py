from typing import List, Tuple, Union

from optuna import Trial

from deepmol.base import Predictor, Transformer
from deepmol.compound_featurization import MolGraphConvFeat, PagtnMolGraphFeat
from deepmol.models.deepchem_model_builders import gat_model, gcn_model, attentivefp_model, pagtn_model, mpnn_model, \
    megnet_model, cnn_model, multitask_classifier_model, multitask_irv_classifier_model, multitask_regressor_model, \
    progressive_multitask_classifier_model, progressive_multitask_regressor_model, robust_multitask_classifier_model, \
    robust_multitask_regressor_model, sc_score_model, atomic_conv_model, chem_ception_model, dag_model, \
    graph_conv_model, smiles_to_vec_model, text_cnn_model, dtnn_model, weave_model, mat_model


# TODO: add support to gpu dgl (pip install  dgl -f https://data.dgl.ai/wheels/cu116/repo.html)
#  and (pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html)


def gat_model_steps(trial: Trial,
                    model_dir: str = None,
                    gat_kwargs: dict = None,
                    deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    featurizer = MolGraphConvFeat()
    # model
    n_attention_heads = trial.suggest_int('n_attention_heads', 4, 10, step=2)
    gat_kwargs['n_attention_heads'] = n_attention_heads
    agg_modes = trial.suggest_categorical('agg_modes', [['mean'], ['flatten']])
    gat_kwargs['agg_modes'] = agg_modes
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    gat_kwargs['dropout'] = dropout
    predictor_dropout = trial.suggest_float('predictor_dropout', 0.0, 0.5, step=0.25)
    gat_kwargs['predictor_dropout'] = predictor_dropout
    model = gat_model(model_dir=model_dir, gat_kwargs=gat_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def gcn_model_steps(trial: Trial,
                    model_dir: str = None,
                    gcn_kwargs: dict = None,
                    deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    featurizer = MolGraphConvFeat()
    # model
    graph_conv_layers = trial.suggest_categorical('graph_conv_layers', [[32, 64], [64, 64], [64, 128]])
    gcn_kwargs['graph_conv_layers'] = graph_conv_layers
    batchnorm = trial.suggest_categorical('batchnorm', [True, False])
    gcn_kwargs['batchnorm'] = batchnorm
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    gcn_kwargs['dropout'] = dropout
    predictor_dropout = trial.suggest_float('predictor_dropout', 0.0, 0.5, step=0.25)
    gcn_kwargs['predictor_dropout'] = predictor_dropout
    model = gcn_model(model_dir=model_dir, gcn_kwargs=gcn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def attentive_fp_model_steps(trial: Trial,
                             model_dir: str = None,
                             attentive_fp_kwargs: dict = None,
                             deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    featurizer = MolGraphConvFeat(use_edges=True)
    # model
    num_layers = trial.suggest_int('num_layers', 1, 5)
    attentive_fp_kwargs['num_layers'] = num_layers
    graph_feat_size = trial.suggest_int('graph_feat_size', 100, 500, step=100)
    attentive_fp_kwargs['graph_feat_size'] = graph_feat_size
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    attentive_fp_kwargs['dropout'] = dropout
    model = attentivefp_model(model_dir=model_dir, attentivefp_kwargs=attentive_fp_kwargs,
                              deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def pagtn_model_steps(trial: Trial,
                      model_dir: str = None,
                      patgn_kwargs: dict = None,
                      deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier/ Regressor
    # PagtnMolGraphFeaturizer
    featurizer = PagtnMolGraphFeat()
    # model
    num_layers = trial.suggest_int('num_layers', 2, 5)
    patgn_kwargs['num_layers'] = num_layers
    num_heads = trial.suggest_int('num_heads', 1, 2)
    patgn_kwargs['num_heads'] = num_heads
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    patgn_kwargs['dropout'] = dropout
    model = pagtn_model(model_dir=model_dir, patgn_kwargs=patgn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def mpnn_model_steps(trial: Trial,
                     model_dir: str = None,
                     mpnn_kwargs: dict = None,
                     deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    featurizer = MolGraphConvFeat(use_edges=True)
    n_hidden = trial.suggest_int('n_hidden', 50, 250, step=50)
    mpnn_kwargs['n_hidden'] = n_hidden
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    mpnn_kwargs['dropout'] = dropout
    model = mpnn_model(model_dir=model_dir, mpnn_kwargs=mpnn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def megnet_model_steps(trial: Trial,
                       model_dir: str = None,
                       megnet_kwargs: dict = None,
                       deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # TODO: add "pip install torch_geometric" to requirements.txt
    # Classifier/ Regressor
    # MolGraphConvFeat
    featurizer = MolGraphConvFeat()
    # model
    n_blocks = trial.suggest_int('n_blocks', 1, 3)
    megnet_kwargs['n_blocks'] = n_blocks
    model = megnet_model(model_dir=model_dir, megnet_kwargs=megnet_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def cnn_model_steps(trial: Trial,
                    model_dir: str = None,
                    cnn_kwargs: dict = None,
                    deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier/ Regressor
    # works with 1D, 2D and 3D data
    layer_filters = trial.suggest_categorical('layer_filters', [[100], [100, 100], [100, 100, 100]])
    cnn_kwargs['layer_filters'] = layer_filters
    kernel_size = trial.suggest_int('kernel_size', 3, 6)
    cnn_kwargs['kernel_size'] = kernel_size
    dropouts = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    cnn_kwargs['dropouts'] = dropouts
    model = cnn_model(model_dir=model_dir, cnn_kwargs=cnn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def multitask_classifier_model_steps(trial: Trial,
                                     model_dir: str = None,
                                     multitask_classifier_kwargs: dict = None,
                                     deepchem_kwargs: dict = None) -> Tuple[str, Predictor]:
    # Classifier
    # 1D descriptors
    dropouts = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    multitask_classifier_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes', [[50], [100], [500], [200, 100]])
    multitask_classifier_kwargs['layer_sizes'] = layer_sizes
    model = multitask_classifier_model(model_dir=model_dir, multitask_classifier_kwargs=multitask_classifier_kwargs,
                                       deepchem_kwargs=deepchem_kwargs)
    return 'model', model


def multitask_irv_classifier_model_steps(trial: Trial,
                                         model_dir: str = None,
                                         multitask_irv_classifier_kwargs: dict = None,
                                         deepchem_kwargs: dict = None) -> Tuple[str, Predictor]:
    # Classifier
    # 1D Descriptors
    K = trial.suggest_int('K', 5, 25, step=5)
    multitask_irv_classifier_kwargs['K'] = K
    model = multitask_irv_classifier_model(model_dir=model_dir,
                                           multitask_irv_classifier_kwargs=multitask_irv_classifier_kwargs,
                                           deepchem_kwargs=deepchem_kwargs)
    return 'model', model


def progressive_multitask_classifier_model_steps(trial: Trial,
                                                 model_dir: str = None,
                                                 progressive_multitask_classifier_kwargs: dict = None,
                                                 deepchem_kwargs: dict = None) -> Tuple[str, Predictor]:
    # Classifier
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    progressive_multitask_classifier_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes', [[50], [100], [500], [200, 100]])
    progressive_multitask_classifier_kwargs['layer_sizes'] = layer_sizes
    model = progressive_multitask_classifier_model(model_dir=model_dir,
                                                   progressive_multitask_classifier_kwargs=progressive_multitask_classifier_kwargs,
                                                   deepchem_kwargs=deepchem_kwargs)
    return 'model', model


def robust_multitask_classifier_model_steps(trial: Trial,
                                            model_dir: str = None,
                                            robust_multitask_classifier_kwargs: dict = None,
                                            deepchem_kwargs: dict = None) -> Tuple[str, Predictor]:
    # Classifier
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    robust_multitask_classifier_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes', [[50], [100], [500], [200, 100]])
    robust_multitask_classifier_kwargs['layer_sizes'] = layer_sizes
    bypass_dropouts = trial.suggest_float('bypass_dropout', 0.0, 0.5, step=0.25)
    robust_multitask_classifier_kwargs['bypass_dropouts'] = bypass_dropouts
    model = robust_multitask_classifier_model(model_dir=model_dir,
                                              robust_multitask_classifier_kwargs=robust_multitask_classifier_kwargs,
                                              deepchem_kwargs=deepchem_kwargs)
    return 'model', model


def sc_score_model_steps(trial: Trial, model_dir: str = None, sc_score_kwargs: dict = None,
                         deepchem_kwargs: dict = None) -> Tuple[str, Predictor]:
    # Classifier
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    sc_score_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes', [[100, 100, 100], [300, 300, 300], [500, 200, 100]])
    sc_score_kwargs['layer_sizes'] = layer_sizes
    model = sc_score_model(model_dir=model_dir, sc_score_kwargs=sc_score_kwargs, deepchem_kwargs=deepchem_kwargs)
    return 'model', model


def atomic_conv_model_steps(trial: Trial, model_dir: str = None, atomic_conv_kwargs: dict = None,
                            deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Classifier/ Regressor
    # ComplexNeighborListFragmentAtomicCoordinates
    featurizer = None  # TODO: define featurizer # TODO: CONTINUE HERE
    model = atomic_conv_model(model_dir=model_dir, atomic_conv_kwargs=atomic_conv_kwargs,
                              deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def chem_ception_model_steps(trial: Trial, model_dir: str = None, chem_ception_kwargs: dict = None,
                             deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Classifier/ Regressor
    # SmilesToImage
    featurizer = None  # TODO: define featurizer
    model = chem_ception_model(model_dir=model_dir, chem_ception_kwargs=chem_ception_kwargs,
                               deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def dag_model_steps(trial: Trial, model_dir: str = None, dag_kwargs: dict = None,
                    deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Classifier/ Regressor
    # ConvMolFeaturizer
    featurizer = None  # TODO: define featurizer
    model = dag_model(model_dir=model_dir, dag_kwargs=dag_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def graph_conv_model_steps(trial: Trial, model_dir: str = None, graph_conv_kwargs: dict = None,
                           deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Classifier/ Regressor
    # ConvMolFeaturizer
    featurizer = None  # TODO: define featurizer
    model = graph_conv_model(model_dir=model_dir, graph_conv_kwargs=graph_conv_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def smiles_to_vec_model_steps(trial: Trial, model_dir: str = None, smiles_to_vec_kwargs: dict = None,
                              deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Classifier/ Regressor
    # SmilesToSeq
    featurizer = None  # TODO: define featurizer
    model = smiles_to_vec_model(model_dir=model_dir, smiles_to_vec_kwargs=smiles_to_vec_kwargs,
                                deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def text_cnn_model_steps(trial: Trial, model_dir: str = None, text_cnn_kwargs: dict = None,
                         deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Classifier/ Regressor
    featurizer = None  # TODO: define featurizer (or check if many are possible)
    model = text_cnn_model(model_dir=model_dir, text_cnn_kwargs=text_cnn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def weave_model_steps(trial: Trial, model_dir: str = None, weave_kwargs: dict = None,
                      deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Classifier/ Regressor
    # WeaveFeaturizer
    featurizer = None  # TODO: define featurizer
    model = weave_model(model_dir=model_dir, weave_kwargs=weave_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def dtnn_model_steps(trial: Trial, model_dir: str = None, dtnn_kwargs: dict = None,
                     deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Regressor
    # CoulombMatrix
    featurizer = None  # TODO: define featurizer
    model = dtnn_model(model_dir=model_dir, dtnn_kwargs=dtnn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def mat_model_steps(trial: Trial, model_dir: str = None, mat_kwargs: dict = None,
                    deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Regressor
    # MATFeaturizer
    featurizer = None  # TODO: define featurizer
    model = mat_model(model_dir=model_dir, mat_kwargs=mat_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def progressive_multitask_regressor_model_steps(trial: Trial,
                                                model_dir: str = None,
                                                progressive_multitask_regressor_kwargs: dict = None,
                                                deepchem_kwargs: dict = None) -> Tuple[str, Predictor]:
    # Regressor
    # 1D Descriptors
    model = progressive_multitask_regressor_model(model_dir=model_dir,
                                                  progressive_multitask_regressor_kwargs=progressive_multitask_regressor_kwargs,
                                                  deepchem_kwargs=deepchem_kwargs)
    return 'model', model


def multitask_regressor_model_steps(trial: Trial, model_dir: str = None, multitask_regressor_kwargs: dict = None,
                                    deepchem_kwargs: dict = None) -> Tuple[str, Predictor]:
    # Regressor
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    multitask_regressor_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes', [[50], [100], [500], [200, 100]])
    multitask_regressor_kwargs['layer_sizes'] = layer_sizes
    model = multitask_regressor_model(model_dir=model_dir, multitask_regressor_kwargs=multitask_regressor_kwargs,
                                      deepchem_kwargs=deepchem_kwargs)
    return 'model', model


def robust_multitask_regressor_model_steps(trial: Trial,
                                           model_dir: str = None,
                                           robust_multitask_regressor_kwargs: dict = None,
                                           deepchem_kwargs: dict = None) -> Tuple[str, Predictor]:
    # Regressor
    # 1D Descriptors
    model = robust_multitask_regressor_model(model_dir=model_dir,
                                             robust_multitask_regressor_kwargs=robust_multitask_regressor_kwargs,
                                             deepchem_kwargs=deepchem_kwargs)
    return 'model', model
