from typing import List, Tuple, Union

from optuna import Trial

from deepmol.base import Predictor, Transformer
from deepmol.compound_featurization import MolGraphConvFeat, PagtnMolGraphFeat, SmileImageFeat, ConvMolFeat, \
    DagTransformer, SmilesSeqFeat, WeaveFeat, DMPNNFeat, MATFeat
from deepmol.models.deepchem_model_builders import gat_model, gcn_model, attentivefp_model, pagtn_model, mpnn_model, \
    megnet_model, cnn_model, multitask_classifier_model, multitask_irv_classifier_model, multitask_regressor_model, \
    progressive_multitask_classifier_model, progressive_multitask_regressor_model, robust_multitask_classifier_model, \
    robust_multitask_regressor_model, sc_score_model, chem_ception_model, dag_model, graph_conv_model, \
    smiles_to_vec_model, text_cnn_model, dtnn_model, weave_model, mat_model, dmpnn_model


# TODO: add support to gpu dgl (pip install  dgl -f https://data.dgl.ai/wheels/cu116/repo.html)
#  and (pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html)


def gat_model_steps(trial: Trial,
                    model_dir: str = 'gat_model/',
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
                    model_dir: str = 'gcn_model/',
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
                             model_dir: str = 'attentive_fp_model/',
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
                      model_dir: str = 'pagtn_model/',
                      pagtn_kwargs: dict = None,
                      deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier/ Regressor
    # PagtnMolGraphFeaturizer
    featurizer = PagtnMolGraphFeat()
    # model
    num_layers = trial.suggest_int('num_layers', 2, 5)
    pagtn_kwargs['num_layers'] = num_layers
    num_heads = trial.suggest_int('num_heads', 1, 2)
    pagtn_kwargs['num_heads'] = num_heads
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    pagtn_kwargs['dropout'] = dropout
    model = pagtn_model(model_dir=model_dir, patgn_kwargs=pagtn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def mpnn_model_steps(trial: Trial,
                     model_dir: str = 'mpnn_model/',
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
                       model_dir: str = 'megnet_model/',
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


def dmpnn_model_steps(trial: Trial,
                      model_dir: str = 'dmpnn_model/',
                      dmpnn_kwargs: dict = None,
                      deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier/ Regressor
    # DMPNNFeaturizer
    featurizer = DMPNNFeat()
    fnn_layers = trial.suggest_int('fnn_layers', 1, 3)
    dmpnn_kwargs['fnn_layers'] = fnn_layers
    fnn_dropout_p = trial.suggest_float('fnn_dropout_p', 0.0, 0.5, step=0.25)
    dmpnn_kwargs['fnn_dropout_p'] = fnn_dropout_p
    depth = trial.suggest_int('depth', 2, 4)
    dmpnn_kwargs['depth'] = depth
    model = dmpnn_model(model_dir=model_dir, dmpnn_kwargs=dmpnn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def cnn_model_steps(trial: Trial,
                    model_dir: str = 'cnn_model/',
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
                                     model_dir: str = 'multitask_classifier_model/',
                                     multitask_classifier_kwargs: dict = None,
                                     deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    # Classifier
    # 1D descriptors
    dropouts = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    multitask_classifier_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes', [[50], [100], [500], [200, 100]])
    multitask_classifier_kwargs['layer_sizes'] = layer_sizes
    model = multitask_classifier_model(model_dir=model_dir, multitask_classifier_kwargs=multitask_classifier_kwargs,
                                       deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def multitask_irv_classifier_model_steps(trial: Trial,
                                         model_dir: str = 'multitask_irv_classifier_model/',
                                         multitask_irv_classifier_kwargs: dict = None,
                                         deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    # Classifier
    # 1D Descriptors
    K = trial.suggest_int('K', 5, 25, step=5)
    multitask_irv_classifier_kwargs['K'] = K
    model = multitask_irv_classifier_model(model_dir=model_dir,
                                           multitask_irv_classifier_kwargs=multitask_irv_classifier_kwargs,
                                           deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def progressive_multitask_classifier_model_steps(trial: Trial,
                                                 model_dir: str = 'progressive_multitask_classifier_model/',
                                                 progressive_multitask_classifier_kwargs: dict = None,
                                                 deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    # Classifier
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    progressive_multitask_classifier_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes', [[50], [100], [500], [200, 100]])
    progressive_multitask_classifier_kwargs['layer_sizes'] = layer_sizes
    model = progressive_multitask_classifier_model(model_dir=model_dir,
                                                   progressive_multitask_classifier_kwargs=progressive_multitask_classifier_kwargs,
                                                   deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def robust_multitask_classifier_model_steps(trial: Trial,
                                            model_dir: str = 'robust_multitask_classifier_model/',
                                            robust_multitask_classifier_kwargs: dict = None,
                                            deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
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
    return [('model', model)]


def sc_score_model_steps(trial: Trial, model_dir: str = 'sc_score_model/', sc_score_kwargs: dict = None,
                         deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    # Classifier
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    sc_score_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes', [[100, 100, 100], [300, 300, 300], [500, 200, 100]])
    sc_score_kwargs['layer_sizes'] = layer_sizes
    model = sc_score_model(model_dir=model_dir, sc_score_kwargs=sc_score_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def chem_ception_model_steps(trial: Trial, model_dir: str = 'chem_ception_model/', chem_ception_kwargs: dict = None,
                             deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Classifier/ Regressor
    # SmilesToImage
    featurizer = SmileImageFeat()
    base_filters = trial.suggest_categorical('base_filters', [8, 16, 32, 64])
    chem_ception_kwargs['base_filters'] = base_filters
    model = chem_ception_model(model_dir=model_dir, chem_ception_kwargs=chem_ception_kwargs,
                               deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def dag_model_steps(trial: Trial, model_dir: str = 'dag_model/', dag_kwargs: dict = None,
                    deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Classifier/ Regressor
    # ConvMolFeaturizer
    featurizer = ConvMolFeat()
    layer_sizes = trial.suggest_categorical('layer_sizes', [[50], [100], [500], [200, 100]])
    dag_kwargs['layer_sizes'] = layer_sizes
    transformer = DagTransformer()
    model = dag_model(model_dir=model_dir, dag_kwargs=dag_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('transformer', transformer), ('model', model)]


def graph_conv_model_steps(trial: Trial, model_dir: str = 'graph_conv_model/', graph_conv_kwargs: dict = None,
                           deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Classifier/ Regressor
    # ConvMolFeaturizer
    featurizer = ConvMolFeat()
    graph_conv_layers = trial.suggest_categorical('graph_conv_layers_conv_model', [[64, 64],
                                                                                   [128, 64],
                                                                                   [256, 128],
                                                                                   [256, 128, 64]])
    graph_conv_kwargs['graph_conv_layers'] = graph_conv_layers
    dense_layer_size = trial.suggest_categorical('dense_layer_size', [128, 256, 512])
    graph_conv_kwargs['dense_layer_size'] = dense_layer_size
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    graph_conv_kwargs['dropout'] = dropout
    model = graph_conv_model(model_dir=model_dir, graph_conv_kwargs=graph_conv_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def smiles_to_vec_model_steps(trial: Trial, model_dir: str = 'smiles_to_vec_model/', smiles_to_vec_kwargs: dict = None,
                              deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Classifier/ Regressor
    # SmilesToSeq
    featurizer = SmilesSeqFeat()
    embedding_dim = trial.suggest_categorical('embedding_dim', [32, 64, 128])
    smiles_to_vec_kwargs['embedding_dim'] = embedding_dim
    filters = trial.suggest_categorical('filters', [32, 64, 128])
    smiles_to_vec_kwargs['filters'] = filters
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
    smiles_to_vec_kwargs['kernel_size'] = kernel_size
    strides = trial.suggest_categorical('strides', [1, 2, 3])
    smiles_to_vec_kwargs['strides'] = strides
    model = smiles_to_vec_model(model_dir=model_dir, smiles_to_vec_kwargs=smiles_to_vec_kwargs,
                                deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def text_cnn_model_steps(trial: Trial, model_dir: str = 'text_cnn_model/', text_cnn_kwargs: dict = None,
                         deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    # Classifier/ Regressor
    n_embedding = trial.suggest_categorical('n_embedding', [50, 75, 100])
    text_cnn_kwargs['n_embedding'] = n_embedding
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    text_cnn_kwargs['dropout'] = dropout
    model = text_cnn_model(model_dir=model_dir, text_cnn_kwargs=text_cnn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def weave_model_steps(trial: Trial, model_dir: str = 'weave_model/', weave_kwargs: dict = None,
                      deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Classifier/ Regressor
    # WeaveFeaturizer
    featurizer = WeaveFeat()
    n_hidden = trial.suggest_categorical('n_hidden', [50, 100, 200])
    weave_kwargs['n_hidden'] = n_hidden
    n_graph_feat = trial.suggest_categorical('n_graph_feat', [64, 128, 256])
    weave_kwargs['n_graph_feat'] = n_graph_feat
    n_weave = trial.suggest_categorical('n_weave', [1, 2, 3])
    weave_kwargs['n_weave'] = n_weave
    dropouts = trial.suggest_float('dropouts', 0.0, 0.5, step=0.25)
    weave_kwargs['dropouts'] = dropouts
    model = weave_model(model_dir=model_dir, weave_kwargs=weave_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def dtnn_model_steps(trial: Trial, model_dir: str = 'dtnn_model/', dtnn_kwargs: dict = None,
                     deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
    # Regressor
    # CoulombMatrix
    n_embedding = trial.suggest_categorical('n_embedding', [50, 75, 100])
    dtnn_kwargs['n_embedding'] = n_embedding
    n_hidden = trial.suggest_categorical('n_hidden', [50, 100, 200])
    dtnn_kwargs['n_hidden'] = n_hidden
    dropout = trial.suggest_float('dropouts', 0.0, 0.5, step=0.25)
    dtnn_kwargs['dropout'] = dropout
    model = dtnn_model(model_dir=model_dir, dtnn_kwargs=dtnn_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def mat_model_steps(trial: Trial, model_dir: str = 'mat_model/', mat_kwargs: dict = None,
                    deepchem_kwargs: dict = None) -> List[Tuple[str, Union[Transformer, Predictor]]]:
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
    model = mat_model(model_dir=model_dir, mat_kwargs=mat_kwargs, deepchem_kwargs=deepchem_kwargs)
    return [('featurizer', featurizer), ('model', model)]


def progressive_multitask_regressor_model_steps(trial: Trial,
                                                model_dir: str = 'progressive_multitask_regressor_model/',
                                                progressive_multitask_regressor_kwargs: dict = None,
                                                deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    # Regressor
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    progressive_multitask_regressor_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes', [[50], [100], [500], [200, 100]])
    progressive_multitask_regressor_kwargs['layer_sizes'] = layer_sizes
    model = progressive_multitask_regressor_model(model_dir=model_dir,
                                                  progressive_multitask_regressor_kwargs=progressive_multitask_regressor_kwargs,
                                                  deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def multitask_regressor_model_steps(trial: Trial, model_dir: str = 'multitask_regressor_model/',
                                    multitask_regressor_kwargs: dict = None,
                                    deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    # Regressor
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    multitask_regressor_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes', [[50], [100], [500], [200, 100]])
    multitask_regressor_kwargs['layer_sizes'] = layer_sizes
    model = multitask_regressor_model(model_dir=model_dir, multitask_regressor_kwargs=multitask_regressor_kwargs,
                                      deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]


def robust_multitask_regressor_model_steps(trial: Trial,
                                           model_dir: str = 'robust_multitask_regressor_model/',
                                           robust_multitask_regressor_kwargs: dict = None,
                                           deepchem_kwargs: dict = None) -> List[Tuple[str, Predictor]]:
    # Regressor
    # 1D Descriptors
    dropouts = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    robust_multitask_regressor_kwargs['dropouts'] = dropouts
    layer_sizes = trial.suggest_categorical('layer_sizes', [[50], [100], [500], [200, 100]])
    robust_multitask_regressor_kwargs['layer_sizes'] = layer_sizes
    bypass_dropouts = trial.suggest_float('bypass_dropout', 0.0, 0.5, step=0.25)
    robust_multitask_regressor_kwargs['bypass_dropouts'] = bypass_dropouts
    model = robust_multitask_regressor_model(model_dir=model_dir,
                                             robust_multitask_regressor_kwargs=robust_multitask_regressor_kwargs,
                                             deepchem_kwargs=deepchem_kwargs)
    return [('model', model)]
