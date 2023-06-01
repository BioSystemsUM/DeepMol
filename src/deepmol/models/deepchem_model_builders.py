from deepchem.models import GATModel, GCNModel, AttentiveFPModel, PagtnModel, MPNNModel, MEGNetModel, DMPNNModel, CNN, \
    MultitaskClassifier, MultitaskIRVClassifier, MultitaskRegressor, ProgressiveMultitaskClassifier, \
    ProgressiveMultitaskRegressor, RobustMultitaskClassifier, RobustMultitaskRegressor, ScScoreModel, AtomicConvModel, \
    ChemCeption, DAGModel, GraphConvModel, Smiles2Vec, TextCNNModel, DTNNModel, WeaveModel
from deepchem.models.torch_models import MATModel

from deepmol.models import DeepChemModel


def gat_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    model = GATModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def gcn_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    model = GCNModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def attentivefp_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    model = AttentiveFPModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def pagtn_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    # PagtnMolGraphFeaturizer MolGraphConvFeaturizer
    model = PagtnModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def mpnn_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    model = MPNNModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def megnet_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    model = MEGNetModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def mat_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Regressor
    # MATFeaturizer
    model = MATModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def dmpnn_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    # DMPNNFeaturizer
    model = DMPNNModel(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def cnn_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier/ Regressor
    model = CNN(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


def multitask_classifier_model(model_dir: str = None, **kwargs) -> DeepChemModel:
    # Classifier
    # CircularFingerprint RDKitDescriptors CoulombMatrixEig RdkitGridFeaturizer BindingPocketFeaturizer
    # ElementPropertyFingerprint
    model = MultitaskClassifier(**kwargs)
    return DeepChemModel(model=model, model_dir=model_dir)


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
