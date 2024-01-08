from typing import List, Tuple, Union

from optuna import Trial

from deepmol.base import Predictor, Transformer
from deepmol.compound_featurization import MolGraphConvFeat, PagtnMolGraphFeat, SmileImageFeat, ConvMolFeat, \
    DagTransformer, SmilesSeqFeat, WeaveFeat, DMPNNFeat, MATFeat
from deepmol.models.deepchem_model_builders import *
from deepchem.feat import PagtnMolGraphFeaturizer
from rdkit.Chem import MolFromSmiles 

def gat_model_steps(trial: Trial, gat_kwargs: dict = None,
                    deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    """
    Steps to optimize a GAT model with optuna.
    It defines the featurizer (MolGraphConvFeat) and the model (GATModel) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    gat_kwargs: dict
        GATModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Union[Predictor, Transformer]]]
        List of tuples (steps) with the featurizer and the model.
    """
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    use_edges = trial.suggest_categorical('use_edges_gat', [True, False])
    use_chirality = trial.suggest_categorical('use_chirality_gat', [True, False])
    use_partial_charge = trial.suggest_categorical('use_partial_charge_gat', [True, False])
    # get number of trues in the list
    n_features = 30
    if use_chirality:
        n_features += 2
    if use_partial_charge:
        n_features += 1
    gat_kwargs['number_atom_features'] = n_features
    featurizer = MolGraphConvFeat(use_edges=use_edges, use_chirality=use_chirality, use_partial_charge=use_partial_charge)

    # model
    n_attention_heads = trial.suggest_int('n_attention_heads', 4, 10, step=2)
    gat_kwargs['n_attention_heads'] = n_attention_heads
    agg_modes = trial.suggest_categorical('agg_modes', [['mean'], ['flatten']])
    gat_kwargs['agg_modes'] = agg_modes
    dropout = trial.suggest_float('dropout_gat', 0.0, 0.5, step=0.25)
    gat_kwargs['dropout'] = dropout
    predictor_dropout = trial.suggest_float('predictor_dropout_gat', 0.0, 0.5, step=0.25)
    gat_kwargs['predictor_dropout'] = predictor_dropout
    model = gat_model(gat_kwargs=gat_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def gcn_model_steps(trial: Trial, gcn_kwargs: dict = None,
                    deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    """
    Steps to optimize a GCN model with optuna.
    It defines the featurizer (MolGraphConvFeat) and the model (GCNModel) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    gcn_kwargs: dict
        GCNModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Union[Predictor, Transformer]]]
        List of tuples (steps) with the featurizer and the model.
    """
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    use_edges = trial.suggest_categorical('use_edges_gcn', [True, False])
    use_chirality = trial.suggest_categorical('use_chirality_gcn', [True, False])
    use_partial_charge = trial.suggest_categorical('use_partial_charge_gcn', [True, False])
    # get number of trues in the list
    n_features = 30
    if use_chirality:
        n_features += 2
    if use_partial_charge:
        n_features += 1
    gcn_kwargs['number_atom_features'] = n_features
    featurizer = MolGraphConvFeat(use_edges=use_edges, use_chirality=use_chirality, use_partial_charge=use_partial_charge)
    # model
    graph_conv_layers = trial.suggest_categorical('graph_conv_layers_gcn',
                                                  [str(cat) for cat in [[32, 64], [64, 64], [64, 128]]])
    gcn_kwargs['graph_conv_layers'] = eval(graph_conv_layers)
    batchnorm = trial.suggest_categorical('batchnorm', [True, False])
    gcn_kwargs['batchnorm'] = batchnorm
    dropout = trial.suggest_float('dropout_gcn', 0.0, 0.5, step=0.25)
    gcn_kwargs['dropout'] = dropout
    predictor_dropout = trial.suggest_float('predictor_dropout_gcn', 0.0, 0.5, step=0.25)
    gcn_kwargs['predictor_dropout'] = predictor_dropout
    model = gcn_model(gcn_kwargs=gcn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def attentive_fp_model_steps(trial: Trial, attentive_fp_kwargs: dict = None,
                             deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    """
    Steps to optimize a AttentiveFP model with optuna.
    It defines the featurizer (MolGraphConvFeat) and the model (AttentiveFPModel) with respective optimizable
    parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    attentive_fp_kwargs: dict
        AttentiveFPModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Union[Predictor, Transformer]]]
        List of tuples (steps) with the featurizer and the model.
    """
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    use_chirality = trial.suggest_categorical('use_chirality_attentive_fp', [True, False])
    use_partial_charge = trial.suggest_categorical('use_partial_charge_attentive_fp', [True, False])
    # get number of trues in the list
    n_features = 30
    if use_chirality:
        n_features += 2
    if use_partial_charge:
        n_features += 1
    attentive_fp_kwargs['number_atom_features'] = n_features
    featurizer = MolGraphConvFeat(use_edges=True, use_chirality=use_chirality, use_partial_charge=use_partial_charge)
    # model
    num_layers = trial.suggest_int('num_layers_attentive_fp', 1, 5)
    attentive_fp_kwargs['num_layers'] = num_layers
    graph_feat_size = trial.suggest_int('graph_feat_size_attentive_fp', 100, 500, step=100)
    attentive_fp_kwargs['graph_feat_size'] = graph_feat_size
    dropout = trial.suggest_float('dropout_attentive_fp', 0.0, 0.5, step=0.25)
    attentive_fp_kwargs['dropout'] = dropout
    model = attentivefp_model( attentivefp_kwargs=attentive_fp_kwargs,
                              deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def pagtn_model_steps(trial: Trial, pagtn_kwargs: dict = None,
                      deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    """
    Steps to optimize a PAGTN model with optuna.
    It defines the featurizer (PagtnMolGraphFeat) and the model (PAGTNModel) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    pagtn_kwargs: dict
        PAGTNModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Union[Predictor, Transformer]]]
        List of tuples (steps) with the featurizer and the model.
    """
    # Classifier/ Regressor
    # PagtnMolGraphFeaturizer
    max_length = trial.suggest_int('max_length_pagtn', 5, 20)
    featurizer = PagtnMolGraphFeat(max_length)
    feature = PagtnMolGraphFeaturizer(max_length=max_length).featurize([MolFromSmiles('CCC')])[0]
    node_features = feature.node_features.shape[1]
    edge_features = feature.edge_features.shape[1]
    # model
    pagtn_kwargs["number_atom_features"] = node_features
    pagtn_kwargs["number_bond_features"] = edge_features
    num_layers = trial.suggest_int('num_layers_pagtn', 2, 5)
    pagtn_kwargs['num_layers'] = num_layers
    num_heads = trial.suggest_int('num_heads', 1, 2)
    pagtn_kwargs['num_heads'] = num_heads
    dropout = trial.suggest_float('dropout_pagtn', 0.0, 0.5, step=0.25)
    pagtn_kwargs['dropout'] = dropout
    model = pagtn_model(patgn_kwargs=pagtn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def mpnn_model_steps(trial: Trial, mpnn_kwargs: dict = None,
                     deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    """
    Steps to optimize a MPNN model with optuna.
    It defines the featurizer (MolGraphConvFeat) and the model (MPNNModel) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    mpnn_kwargs: dict
        MPNNModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Union[Predictor, Transformer]]]
        List of tuples (steps) with the featurizer and the model.
    """
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    use_chirality = trial.suggest_categorical('use_chirality_mpnn', [True, False])
    use_partial_charge = trial.suggest_categorical('use_partial_charge_mpnn', [True, False])
    # get number of trues in the list
    n_features = 30
    if use_chirality:
        n_features += 2
    if use_partial_charge:
        n_features += 1
    featurizer = MolGraphConvFeat(use_edges=True, use_chirality=use_chirality, use_partial_charge=use_partial_charge)
    n_hidden = trial.suggest_int('n_hidden_mpnn', 50, 250, step=50)
    mpnn_kwargs['n_hidden'] = n_hidden
    dropout = trial.suggest_float('dropout_mpnn', 0.0, 0.5, step=0.25)
    mpnn_kwargs['dropout'] = dropout
    model = mpnn_model(mpnn_kwargs=mpnn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def megnet_model_steps(trial: Trial, megnet_kwargs: dict = None,
                       deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    """
    Steps to optimize a MEGNET model with optuna.
    It defines the featurizer (MolGraphConvFeat) and the model (MEGNETModel) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    megnet_kwargs: dict
        MEGNETModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Union[Predictor, Transformer]]]
        List of tuples (steps) with the featurizer and the model.
    """
    # Classifier/ Regressor
    # MolGraphConvFeat
    use_edges = trial.suggest_categorical('use_edges_megnet', [True, False])
    use_chirality = trial.suggest_categorical('use_chirality_megnet', [True, False])
    use_partial_charge = trial.suggest_categorical('use_partial_charge_megnet', [True, False])
    # get number of trues in the list
    n_features = 30
    if use_chirality:
        n_features += 2
    if use_partial_charge:
        n_features += 1
    # megnet_kwargs['number_atom_features'] = n_features
    featurizer = MolGraphConvFeat(use_edges=use_edges, use_chirality=use_chirality, use_partial_charge=use_partial_charge)
    # model
    n_blocks = trial.suggest_int('n_blocks', 1, 3)
    megnet_kwargs['n_blocks'] = n_blocks
    model = megnet_model(megnet_kwargs=megnet_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def dmpnn_model_steps(trial: Trial, dmpnn_kwargs: dict = None,
                      deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    """
    Steps to optimize a DMPNN model with optuna.
    It defines the featurizer (DMPNNFeat) and the model (DMPNNModel) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    dmpnn_kwargs: dict
        DMPNNModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Union[Predictor, Transformer]]]
        List of tuples (steps) with the featurizer and the model.
    """
    # Classifier/ Regressor
    # DMPNNFeaturizer
    featurizer = DMPNNFeat()
    fnn_layers = trial.suggest_int('fnn_layers', 1, 3)
    dmpnn_kwargs['fnn_layers'] = fnn_layers
    fnn_dropout_p = trial.suggest_float('fnn_dropout_p', 0.0, 0.5, step=0.25)
    dmpnn_kwargs['fnn_dropout_p'] = fnn_dropout_p
    depth = trial.suggest_int('depth_dmpnn', 2, 4)
    dmpnn_kwargs['depth'] = depth
    model = dmpnn_model(dmpnn_kwargs=dmpnn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def cnn_model_steps(trial: Trial, cnn_kwargs: dict = None,
                    deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    """
    Steps to optimize a CNN model with optuna.
    It defines the model (CNNModel) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    cnn_kwargs: dict
        CNNModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Union[Predictor, Transformer]]]
        List of tuples (steps) with the model.
    """
    # Classifier/ Regressor
    # works with 1D, 2D and 3D data
    layer_filters = trial.suggest_categorical('layer_filters',
                                              [str(cat) for cat in [[100], [100, 100], [100, 100, 100]]])
    cnn_kwargs['layer_filters'] = eval(layer_filters)
    kernel_size = trial.suggest_int('kernel_size_cnn', 3, 6)
    cnn_kwargs['kernel_size'] = kernel_size
    dropouts = trial.suggest_float('dropout_cnn', 0.0, 0.5, step=0.25)
    cnn_kwargs['dropouts'] = dropouts
    model = cnn_model(cnn_kwargs=cnn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def multitask_classifier_model_steps(trial: Trial,
                                     multitask_classifier_kwargs: dict = None,
                                     deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    """
    Steps to optimize a multitask classifier model with optuna.
    It defines the model (MultitaskClassifier) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    multitask_classifier_kwargs: dict
        MultitaskClassifier parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Predictor]]
        List of tuples (steps) with the model.
    """
    # Classifier
    # 1D descriptors
    dropouts = trial.suggest_float('dropout_multitask_classifier', 0.0, 0.5, step=0.25)
    multitask_classifier_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes_multitask_classifier_model', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
    multitask_classifier_kwargs['layer_sizes'] = eval(layer_sizes)
    model = multitask_classifier_model(multitask_classifier_kwargs=multitask_classifier_kwargs,
                                       deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def multitask_irv_classifier_model_steps(trial: Trial,
                                         multitask_irv_classifier_kwargs: dict = None,
                                         deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    """
    Steps to optimize a multitask IRV classifier model with optuna.
    It defines the model (MultitaskIRVClassifier) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    multitask_irv_classifier_kwargs: dict
        MultitaskIRVClassifier parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Predictor]]
        List of tuples (steps) with the model.
    """
    # Classifier
    # 1D Descriptors
    K = trial.suggest_int('K', 5, 25, step=5)
    multitask_irv_classifier_kwargs['K'] = K
    model = multitask_irv_classifier_model(multitask_irv_classifier_kwargs=multitask_irv_classifier_kwargs,
                                           deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def progressive_multitask_classifier_model_steps(trial: Trial,
                                                 progressive_multitask_classifier_kwargs: dict = None,
                                                 deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    """
    Steps to optimize a progressive multitask classifier model with optuna.
    It defines the model (ProgressiveMultitaskClassifier) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    progressive_multitask_classifier_kwargs: dict
        ProgressiveMultitaskClassifier parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Predictor]]
        List of tuples (steps) with the model.
    """
    # Classifier
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout_progressive_multitask', 0.0, 0.5, step=0.25)
    progressive_multitask_classifier_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes_progressive_multitask_classifier', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
    progressive_multitask_classifier_kwargs['layer_sizes'] = eval(layer_sizes)
    model = progressive_multitask_classifier_model(progressive_multitask_classifier_kwargs=progressive_multitask_classifier_kwargs,
                                                   deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def robust_multitask_classifier_model_steps(trial: Trial,
                                            robust_multitask_classifier_kwargs: dict = None,
                                            deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    """
    Steps to optimize a robust multitask classifier model with optuna.
    It defines the model (RobustMultitaskClassifier) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    robust_multitask_classifier_kwargs: dict
        RobustMultitaskClassifier parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Predictor]]
        List of tuples (steps) with the model.
    """
    # Classifier
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
    robust_multitask_classifier_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes_robust_multitask_classifier', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
    robust_multitask_classifier_kwargs['layer_sizes'] = eval(layer_sizes)
    bypass_dropouts = trial.suggest_float('bypass_dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
    robust_multitask_classifier_kwargs['bypass_dropouts'] = bypass_dropouts
    model = robust_multitask_classifier_model(
                                              robust_multitask_classifier_kwargs=robust_multitask_classifier_kwargs,
                                              deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def sc_score_model_steps(trial: Trial, sc_score_kwargs: dict = None,
                         deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    """
    Steps to optimize a sc score model with optuna.
    It defines the model (SCScoreModel) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    sc_score_kwargs: dict
        SCScoreModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Predictor]]
        List of tuples (steps) with the model.
    """
    # Classifier
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout_sc_score', 0.0, 0.5, step=0.25)
    sc_score_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes_sc_score',
                                            [str(cat) for cat in [[100, 100, 100], [300, 300, 300], [500, 200, 100]]])
    sc_score_kwargs['layer_sizes'] = eval(layer_sizes)
    model = sc_score_model(sc_score_kwargs=sc_score_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def chem_ception_model_steps(trial: Trial, chem_ception_kwargs: dict = None,
                             deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    """
    Steps to optimize a chem ception model with optuna.
    It defines the featurizer (SmilesImageFeat) and the model (ChemCeption) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    chem_ception_kwargs: dict
        ChemCeption parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Union[Transformer, Predictor]]]
        List of tuples (steps) with the featurizer and the model.
    """
    # Classifier/ Regressor
    # SmilesToImage
    featurizer = SmileImageFeat()
    base_filters = trial.suggest_categorical('base_filters', [8, 16, 32, 64])
    chem_ception_kwargs['base_filters'] = base_filters
    model = chem_ception_model(chem_ception_kwargs=chem_ception_kwargs,
                               deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def dag_model_steps(trial: Trial, dag_kwargs: dict = None,
                    deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    """
    Steps to optimize a dag model with optuna.
    It defines the featurizer (ConvMolFeat) and the model (DAGModel) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    dag_kwargs: dict
        DAGModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Union[Transformer, Predictor]]]
        List of tuples (steps) with the featurizer and the model.
    """
    # Classifier/ Regressor
    # ConvMolFeaturizer
    featurizer = ConvMolFeat()
    layer_sizes = trial.suggest_categorical('layer_sizes_dag', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
    dag_kwargs['layer_sizes'] = eval(layer_sizes)
    dropout = trial.suggest_float('dropout_dag', 0.0, 0.5, step=0.25)
    dag_kwargs['dropout'] = dropout
    layer_sizes_gather = trial.suggest_categorical('layer_sizes_gather', 
                                                    [str(cat) for cat in [[50], [100], [500], [200, 100]]])
    dag_kwargs['layer_sizes_gather'] = eval(layer_sizes_gather)
    n_graph_feat = trial.suggest_categorical('n_graph_feat_dag', [30, 64, 128, 256])
    dag_kwargs['n_graph_feat'] = n_graph_feat

    transformer = DagTransformer()
    dag_kwargs["n_atom_feat"] = 85
    model = dag_model( dag_kwargs=dag_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('transformer', transformer), ('model', model)]


def graph_conv_model_steps(trial: Trial, graph_conv_kwargs: dict = None,
                           deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    """
    Steps to optimize a graph conv model with optuna.
    It defines the featurizer (ConvMolFeat) and the model (GraphConvModel) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    graph_conv_kwargs: dict
        GraphConvModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Union[Transformer, Predictor]]]
        List of tuples (steps) with the featurizer and the model.
    """
    # Classifier/ Regressor
    # ConvMolFeaturizer
    master_atom = trial.suggest_categorical('master_atom', [True, False])
    use_chirality = trial.suggest_categorical('use_chirality_graph_conv', [True, False])

    featurizer = ConvMolFeat(master_atom=master_atom, use_chirality=use_chirality)
    graph_conv_layers = trial.suggest_categorical('graph_conv_layers_conv_model',
                                                  [str(cat) for cat in [[64, 64], [128, 64],
                                                                        [256, 128], [256, 128, 64]]])
    graph_conv_kwargs['graph_conv_layers'] = eval(graph_conv_layers)
    dense_layer_size = trial.suggest_categorical('dense_layer_size', [128, 256, 512])
    graph_conv_kwargs['dense_layer_size'] = dense_layer_size
    dropout = trial.suggest_float('dropout_graph_conv', 0.0, 0.5, step=0.25)
    graph_conv_kwargs['dropout'] = dropout
    model = graph_conv_model( graph_conv_kwargs=graph_conv_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def smiles_to_vec_model_steps(trial: Trial, smiles_to_vec_kwargs: dict = None,
                              deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    """
    Steps to optimize a smiles to vec model with optuna.
    It defines the featurizer (SmilesSeqFeat) and the model (SmilesToVec) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    smiles_to_vec_kwargs: dict
        SmilesToVec parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Union[Transformer, Predictor]]]
        List of tuples (steps) with the featurizer and the model.
    """
    # Classifier/ Regressor
    # SmilesToSeq
    featurizer = SmilesSeqFeat()
    embedding_dim = trial.suggest_categorical('embedding_dim', [32, 64, 128])
    smiles_to_vec_kwargs['embedding_dim'] = embedding_dim
    filters = trial.suggest_categorical('filters', [32, 64, 128])
    smiles_to_vec_kwargs['filters'] = filters
    kernel_size = trial.suggest_categorical('kernel_size_smiles_to_vec', [3, 5, 7])
    smiles_to_vec_kwargs['kernel_size'] = kernel_size
    strides = trial.suggest_categorical('strides', [1, 2, 3])
    smiles_to_vec_kwargs['strides'] = strides
    model = smiles_to_vec_model( smiles_to_vec_kwargs=smiles_to_vec_kwargs,
                                deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def text_cnn_model_steps(trial: Trial, text_cnn_kwargs: dict = None,
                         deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    """
    Steps to optimize a text cnn model with optuna.
    It defines the model (TextCNNModel) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    text_cnn_kwargs: dict
        TextCNNModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Predictor]]
        List of tuples (steps) with the model.
    """
    # Classifier/ Regressor
    n_embedding = trial.suggest_categorical('n_embedding_text_cnn', [50, 75, 100])
    text_cnn_kwargs['n_embedding'] = n_embedding
    dropout = trial.suggest_float('dropout_text_cnn', 0.0, 0.5, step=0.25)
    text_cnn_kwargs['dropout'] = dropout
    model = text_cnn_model(text_cnn_kwargs=text_cnn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def weave_model_steps(trial: Trial, weave_kwargs: dict = None,
                      deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    """
    Steps to optimize a weave model with optuna.
    It defines the featurizer (WeaveFeat) and the model (WeaveModel) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    weave_kwargs: dict
        WeaveModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Union[Transformer, Predictor]]]
        List of tuples (steps) with the featurizer and the model.
    """
    # Classifier/ Regressor
    # WeaveFeaturizer
    use_chirality = trial.suggest_categorical('use_chirality_weave', [True, False])
    if use_chirality:
        n_atom_feat = 78
        weave_kwargs['n_atom_feat'] = n_atom_feat
        weave_kwargs['n_pair_feat'] = 18
    
    featurizer = WeaveFeat(use_chirality=use_chirality)
    n_hidden = trial.suggest_categorical('n_hidden_weave', [50, 100, 200])
    weave_kwargs['n_hidden'] = n_hidden
    n_graph_feat = trial.suggest_categorical('n_graph_feat_weave', [64, 128, 256])
    weave_kwargs['n_graph_feat'] = n_graph_feat
    # n_weave = trial.suggest_categorical('n_weave', [1, 2, 3])
    # weave_kwargs['n_weave'] = n_weave
    dropouts = trial.suggest_float('dropouts_weave', 0.0, 0.5, step=0.25)
    weave_kwargs['dropouts'] = dropouts
    model = weave_model(weave_kwargs=weave_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def dtnn_model_steps(trial: Trial, dtnn_kwargs: dict = None,
                     deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    """
    Steps to optimize a dtnn model with optuna.
    It defines the model (DTNNModel) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    dtnn_kwargs: dict
        DTNNModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Predictor]]
        List of tuples (steps) with the model.
    """
    # Regressor
    # CoulombMatrix
    n_embedding = trial.suggest_categorical('n_embedding_dtnn', [50, 75, 100])
    dtnn_kwargs['n_embedding'] = n_embedding
    n_hidden = trial.suggest_categorical('n_hidden_dtnn', [50, 100, 200])
    dtnn_kwargs['n_hidden'] = n_hidden
    dropout = trial.suggest_float('dropouts_dtnn', 0.0, 0.5, step=0.25)
    dtnn_kwargs['dropout'] = dropout
    model = dtnn_model(dtnn_kwargs=dtnn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def mat_model_steps(trial: Trial, mat_kwargs: dict = None,
                    deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    """
    Steps to optimize a mat model with optuna.
    It defines the featurizer (MATFeat) and the model (MATModel) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    mat_kwargs: dict
        MATModel parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Union[Transformer, Predictor]]]
        List of tuples (steps) with the featurizer and the model.
    """
    # Regressor
    # MATFeaturizer
    featurizer = MATFeat()
    n_encoders = trial.suggest_int('n_encoders', 4, 10, step=2)
    mat_kwargs['n_encoders'] = n_encoders
    sa_dropout_p = trial.suggest_float('sa_dropout_p', 0.0, 0.5, step=0.25)
    mat_kwargs['sa_dropout_p'] = sa_dropout_p
    n_layers = trial.suggest_int('n_layers', 1, 3)
    mat_kwargs['n_layers'] = n_layers
    ff_dropout_p = trial.suggest_float('ff_dropout_p', 0.0, 0.5, step=0.25)
    mat_kwargs['ff_dropout_p'] = ff_dropout_p
    encoder_dropout_p = trial.suggest_float('encoder_dropout_p', 0.0, 0.5, step=0.25)
    mat_kwargs['encoder_dropout_p'] = encoder_dropout_p
    embed_dropout_p = trial.suggest_float('embed_dropout_p', 0.0, 0.5, step=0.25)
    mat_kwargs['embed_dropout_p'] = embed_dropout_p
    gen_dropout_p = trial.suggest_float('gen_dropout_p', 0.0, 0.5, step=0.25)
    mat_kwargs['gen_dropout_p'] = gen_dropout_p
    gen_n_layers = trial.suggest_int('gen_n_layers', 1, 3)
    mat_kwargs['gen_n_layers'] = gen_n_layers
    model = mat_model(mat_kwargs=mat_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def progressive_multitask_regressor_model_steps(trial: Trial,
                                                progressive_multitask_regressor_kwargs: dict = None,
                                                deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    """
    Steps to optimize a progressive multitask regressor model with optuna.
    It defines the model (ProgressiveMultitaskRegressor) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    progressive_multitask_regressor_kwargs: dict
        ProgressiveMultitaskRegressor parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Predictor]]
        List of tuples (steps) with the model.
    """
    # Regressor
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout_progressive_multitask_regressor', 0.0, 0.5, step=0.25)
    progressive_multitask_regressor_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes_progressive_multitask_regressor', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
    progressive_multitask_regressor_kwargs['layer_sizes'] = eval(layer_sizes)
    model = progressive_multitask_regressor_model(progressive_multitask_regressor_kwargs=progressive_multitask_regressor_kwargs,
                                                  deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def multitask_regressor_model_steps(trial: Trial,
                                    multitask_regressor_kwargs: dict = None,
                                    deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    """
    Steps to optimize a multitask regressor model with optuna.
    It defines the model (MultitaskRegressor) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    multitask_regressor_kwargs: dict
        MultitaskRegressor parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Predictor]]
        List of tuples (steps) with the model.
    """
    # Regressor
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout_multitask_regressor', 0.0, 0.5, step=0.25)
    multitask_regressor_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes_multitask_regressor', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
    multitask_regressor_kwargs['layer_sizes'] = eval(layer_sizes)
    model = multitask_regressor_model(multitask_regressor_kwargs=multitask_regressor_kwargs,
                                      deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def robust_multitask_regressor_model_steps(trial: Trial,
                                           robust_multitask_regressor_kwargs: dict = None,
                                           deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    """
    Steps to optimize a robust multitask regressor model with optuna.
    It defines the model (RobustMultitaskRegressor) with respective optimizable parameters.

    Parameters
    ----------
    trial: optuna.Trial
        Optuna trial object.
    robust_multitask_regressor_kwargs: dict
        RobustMultitaskRegressor parameters.
    deepchem_kwargs: dict
        Deepchem parameters.

    Returns
    -------
    List[Tuple[str, Predictor]]
        List of tuples (steps) with the model.
    """
    # Regressor
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout_robust_multitask_regressor', 0.0, 0.5, step=0.25)
    robust_multitask_regressor_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes_robust_multitask_regressor', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
    robust_multitask_regressor_kwargs['layer_sizes'] = eval(layer_sizes)
    bypass_dropouts = trial.suggest_float('bypass_dropout_robust_multitask_regressor', 0.0, 0.5, step=0.25)
    robust_multitask_regressor_kwargs['bypass_dropouts'] = bypass_dropouts
    model = robust_multitask_regressor_model(robust_multitask_regressor_kwargs=robust_multitask_regressor_kwargs,
                                             deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]
