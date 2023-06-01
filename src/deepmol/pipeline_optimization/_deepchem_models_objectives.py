from typing import List, Tuple, Union

from deepchem.models import GATModel, GCNModel, AttentiveFPModel, PagtnModel, MPNNModel, MEGNetModel, CNN, \
    MultitaskClassifier, MultitaskIRVClassifier, MultitaskRegressor, ProgressiveMultitaskClassifier, \
    ProgressiveMultitaskRegressor, RobustMultitaskClassifier, RobustMultitaskRegressor, ScScoreModel, AtomicConvModel, \
    ChemCeption, DAGModel, GraphConvModel, Smiles2Vec, TextCNNModel, DTNNModel, WeaveModel
from deepchem.models.torch_models import MATModel
from optuna import Trial

from deepmol.base import Predictor, Transformer
from deepmol.compound_featurization import MolGraphConvFeat, PagtnMolGraphFeat
from deepmol.models import DeepChemModel

# TODO: add support to gpu dgl (pip install  dgl -f https://data.dgl.ai/wheels/cu116/repo.html)
#  and (pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html)


def gat_model_steps(trial: Trial, model_dir: str = None, **kwargs) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    featurizer = MolGraphConvFeat()
    # model
    n_attention_heads = trial.suggest_int('n_attention_heads', 4, 10, step=2)
    kwargs['n_attention_heads'] = n_attention_heads
    agg_modes = trial.suggest_categorical('agg_modes', [['mean'], ['flatten']])
    kwargs['agg_modes'] = agg_modes
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    kwargs['dropout'] = dropout
    predictor_dropout = trial.suggest_float('predictor_dropout', 0.0, 0.5, step=0.25)
    kwargs['predictor_dropout'] = predictor_dropout
    model = GATModel(**kwargs)
    model = DeepChemModel(model=model, model_dir=model_dir)
    return [('featurizer', featurizer), ('model', model)]


def gcn_model_steps(trial: Trial, model_dir: str = None, **kwargs) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    featurizer = MolGraphConvFeat()
    # model
    graph_conv_layers = trial.suggest_categorical('graph_conv_layers', [[32, 64], [64, 64], [64, 128]])
    kwargs['graph_conv_layers'] = graph_conv_layers
    batchnorm = trial.suggest_categorical('batchnorm', [True, False])
    kwargs['batchnorm'] = batchnorm
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    kwargs['dropout'] = dropout
    predictor_dropout = trial.suggest_float('predictor_dropout', 0.0, 0.5, step=0.25)
    kwargs['predictor_dropout'] = predictor_dropout
    model = GCNModel(**kwargs)
    model = DeepChemModel(model=model, model_dir=model_dir)
    return [('featurizer', featurizer), ('model', model)]


def attentive_fp_model_steps(trial: Trial,
                             model_dir: str = None,
                             **kwargs) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    featurizer = MolGraphConvFeat(use_edges=True)
    # model
    num_layers = trial.suggest_int('num_layers', 1, 5)
    kwargs['num_layers'] = num_layers
    graph_feat_size = trial.suggest_int('graph_feat_size', 100, 500, step=100)
    kwargs['graph_feat_size'] = graph_feat_size
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    kwargs['dropout'] = dropout
    model = AttentiveFPModel(**kwargs)
    model = DeepChemModel(model=model, model_dir=model_dir)
    return [('featurizer', featurizer), ('model', model)]


def pagtn_model_steps(trial: Trial, model_dir: str = None, **kwargs) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier/ Regressor
    # PagtnMolGraphFeaturizer
    featurizer = PagtnMolGraphFeat()
    # model
    num_layers = trial.suggest_int('num_layers', 2, 5)
    kwargs['num_layers'] = num_layers
    num_heads = trial.suggest_int('num_heads', 1, 2)
    kwargs['num_heads'] = num_heads
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    kwargs['dropout'] = dropout
    model = PagtnModel(**kwargs)
    model = DeepChemModel(model=model, model_dir=model_dir)
    return [('featurizer', featurizer), ('model', model)]


def mpnn_model_steps(trial: Trial, model_dir: str = None, **kwargs) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    featurizer = MolGraphConvFeat(use_edges=True)
    n_hidden = trial.suggest_int('n_hidden', 50, 250, step=50)
    kwargs['n_hidden'] = n_hidden
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    kwargs['dropout'] = dropout
    model = MPNNModel(**kwargs)
    model = DeepChemModel(model=model, model_dir=model_dir)
    return [('featurizer', featurizer), ('model', model)]


def megnet_model_steps(trial: Trial,
                       model_dir: str = None,
                       **kwargs) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # TODO: add "pip install torch_geometric" to requirements.txt
    # Classifier/ Regressor
    # MolGraphConvFeat
    featurizer = MolGraphConvFeat()
    # model
    n_blocks = trial.suggest_int('n_blocks', 1, 3)
    kwargs['n_blocks'] = n_blocks
    model = MEGNetModel(**kwargs)
    model = DeepChemModel(model=model, model_dir=model_dir)
    return [('featurizer', featurizer), ('model', model)]


def cnn_model_steps(trial: Trial, model_dir: str = None, **kwargs) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier/ Regressor
    # works with 1D, 2D and 3D data
    layer_filters = trial.suggest_categorical('layer_filters', [[100], [100, 100], [100, 100, 100]])
    kwargs['layer_filters'] = layer_filters
    kernel_size = trial.suggest_int('kernel_size', 3, 6)
    kwargs['kernel_size'] = kernel_size
    dropouts = trial.suggest_float('dropout', 0.0, 0.5, step=0.25)
    kwargs['dropouts'] = dropouts
    model = CNN(**kwargs)
    model = DeepChemModel(model=model, model_dir=model_dir)
    return [('model', model)]


def multitask_classifier_model_steps(trial: Trial,
                                     model_dir: str = None,
                                     **kwargs) -> List[Tuple[str, Union[Predictor, Transformer]]]:
    # Classifier
    # CircularFingerprint RDKitDescriptors CoulombMatrixEig RdkitGridFeaturizer BindingPocketFeaturizer
    # ElementPropertyFingerprint
    model = MultitaskClassifier(**kwargs)
    model = DeepChemModel(model=model, model_dir=model_dir)
    return [('model', model)]


def multitask_irv_classifier_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier
    # CircularFingerprint RDKitDescriptors CoulombMatrixEig RdkitGridFeaturizer BindingPocketFeaturizer
    # ElementPropertyFingerprint
    model = MultitaskIRVClassifier(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def multitask_regressor_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Regressor
    # CircularFingerprint RDKitDescriptors CoulombMatrixEig RdkitGridFeaturizer BindingPocketFeaturizer
    # ElementPropertyFingerprint
    model = MultitaskRegressor(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def progressive_multitask_classifier_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier
    # CircularFingerprint RDKitDescriptors CoulombMatrixEig RdkitGridFeaturizer BindingPocketFeaturizer
    # ElementPropertyFingerprint
    model = ProgressiveMultitaskClassifier(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def progressive_multitask_regressor_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Regressor
    # CircularFingerprint RDKitDescriptors CoulombMatrixEig RdkitGridFeaturizer BindingPocketFeaturizer
    # ElementPropertyFingerprint
    model = ProgressiveMultitaskRegressor(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def robust_multitask_classifier_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier
    # CircularFingerprint RDKitDescriptors CoulombMatrixEig RdkitGridFeaturizer BindingPocketFeaturizer
    # ElementPropertyFingerprint
    model = RobustMultitaskClassifier(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def robust_multitask_regressor_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Regressor
    # CircularFingerprint RDKitDescriptors CoulombMatrixEig RdkitGridFeaturizer BindingPocketFeaturizer
    # ElementPropertyFingerprint
    model = RobustMultitaskRegressor(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def sc_score_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier
    # CircularFingerprint
    model = ScScoreModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def atomic_conv_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    # ComplexNeighborListFragmentAtomicCoordinates
    model = AtomicConvModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def chem_ception_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    # SmilesToImage
    model = ChemCeption(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def dag_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    # ConvMolFeaturizer
    model = DAGModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def graph_conv_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    # ConvMolFeaturizer
    model = GraphConvModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def smiles_to_vec_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    # SmilesToSeq
    model = Smiles2Vec(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def text_cnn_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    model = TextCNNModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def dtnn_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Regressor
    # CoulombMatrix
    model = DTNNModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def weave_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    # WeaveFeaturizer
    model = WeaveModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def mat_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Regressor
    # MATFeaturizer
    model = MATModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)
