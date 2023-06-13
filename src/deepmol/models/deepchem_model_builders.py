from deepchem.models import GATModel, GCNModel, AttentiveFPModel, PagtnModel, MPNNModel, MEGNetModel, DMPNNModel, CNN, \
    MultitaskClassifier, MultitaskIRVClassifier, MultitaskRegressor, ProgressiveMultitaskClassifier, \
    ProgressiveMultitaskRegressor, RobustMultitaskClassifier, RobustMultitaskRegressor, ScScoreModel, AtomicConvModel, \
    ChemCeption, DAGModel, GraphConvModel, Smiles2Vec, TextCNNModel, DTNNModel, WeaveModel
from deepchem.models.chemnet_layers import Stem, InceptionResnetA, ReductionA, InceptionResnetB, ReductionB, \
    InceptionResnetC
from deepchem.models.layers import DTNNEmbedding, Highway, Stack, DAGLayer, DAGGather, WeaveLayer, WeaveGather
from deepchem.models.torch_models import MATModel
from keras.initializers.initializers import TruncatedNormal

from deepmol.models import DeepChemModel


def gat_model(model_dir: str = 'gat_model/', gat_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    gat_kwargs = gat_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    model = GATModel(**gat_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def gcn_model(model_dir: str = 'gcn_model/', gcn_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    gcn_kwargs = gcn_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    model = GCNModel(**gcn_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def attentivefp_model(model_dir: str = 'attentivefp_model/',
                      attentivefp_kwargs: dict = None,
                      deepchem_kwargs: dict = None) -> DeepChemModel:
    attentivefp_kwargs = attentivefp_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    model = AttentiveFPModel(**attentivefp_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def pagtn_model(model_dir: str = 'pagtn_model/', patgn_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    patgn_kwargs = patgn_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # PagtnMolGraphFeaturizer MolGraphConvFeaturizer
    model = PagtnModel(**patgn_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def mpnn_model(model_dir: str = 'mpnn_model/', mpnn_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    mpnn_kwargs = mpnn_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    model = MPNNModel(**mpnn_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def megnet_model(model_dir: str = 'megnet_model/', megnet_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    megnet_kwargs = megnet_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    model = MEGNetModel(**megnet_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def dmpnn_model(model_dir: str = 'dmpnn_model/', dmpnn_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    dmpnn_kwargs = dmpnn_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # DMPNNFeaturizer
    model = DMPNNModel(**dmpnn_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def cnn_model(model_dir: str = 'cnn_model/', cnn_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    cnn_kwargs = cnn_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    model = CNN(**cnn_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def multitask_classifier_model(model_dir: str = 'multitask_classifier_model/', multitask_classifier_kwargs: dict = None,
                               deepchem_kwargs: dict = None) -> DeepChemModel:
    multitask_classifier_kwargs = multitask_classifier_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier
    # 1D Descriptors
    model = MultitaskClassifier(**multitask_classifier_kwargs)
    model.mode = 'classification'
    model.model.mode = 'classification'
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def multitask_irv_classifier_model(model_dir: str = 'multitask_irv_classifier_model/',
                                   multitask_irv_classifier_kwargs: dict = None,
                                   deepchem_kwargs: dict = None) -> DeepChemModel:
    multitask_irv_classifier_kwargs = multitask_irv_classifier_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier
    # 1D Descriptors
    model = MultitaskIRVClassifier(**multitask_irv_classifier_kwargs)
    model.model.mode = 'classification'
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def progressive_multitask_classifier_model(model_dir: str = 'progressive_multitask_classifier_model/',
                                           progressive_multitask_classifier_kwargs: dict = None,
                                           deepchem_kwargs: dict = None) -> DeepChemModel:
    progressive_multitask_classifier_kwargs = progressive_multitask_classifier_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier
    # 1D Descriptors
    model = ProgressiveMultitaskClassifier(**progressive_multitask_classifier_kwargs)
    model.model.mode = 'classification'
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def progressive_multitask_regressor_model(model_dir: str = 'progressive_multitask_regressor_model/',
                                          progressive_multitask_regressor_kwargs: dict = None,
                                          deepchem_kwargs: dict = None) -> DeepChemModel:
    progressive_multitask_regressor_kwargs = progressive_multitask_regressor_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Regressor
    # CircularFingerprint RDKitDescriptors CoulombMatrixEig RdkitGridFeaturizer BindingPocketFeaturizer
    # ElementPropertyFingerprint
    model = ProgressiveMultitaskRegressor(**progressive_multitask_regressor_kwargs)
    model.model.mode = 'regression'
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def robust_multitask_classifier_model(model_dir: str = 'robust_multitask_classifier_model/',
                                      robust_multitask_classifier_kwargs: dict = None,
                                      deepchem_kwargs: dict = None) -> DeepChemModel:
    robust_multitask_classifier_kwargs = robust_multitask_classifier_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier
    # 1D Descriptors
    model = RobustMultitaskClassifier(**robust_multitask_classifier_kwargs)
    model.model.mode = 'classification'
    custom_objects = {'Stack': Stack}
    return DeepChemModel(model=model, model_dir=model_dir, custom_objects=custom_objects, **deepchem_kwargs)


def robust_multitask_regressor_model(model_dir: str = 'robust_multitask_regressor_model/',
                                     robust_multitask_regressor_kwargs: dict = None,
                                     deepchem_kwargs: dict = None) -> DeepChemModel:
    robust_multitask_regressor_kwargs = robust_multitask_regressor_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Regressor
    # CircularFingerprint RDKitDescriptors CoulombMatrixEig RdkitGridFeaturizer BindingPocketFeaturizer
    # ElementPropertyFingerprint
    model = RobustMultitaskRegressor(**robust_multitask_regressor_kwargs)
    model.model.mode = 'regression'
    custom_objects = {'Stack': Stack}
    return DeepChemModel(model=model, model_dir=model_dir, custom_objects=custom_objects, **deepchem_kwargs)


def sc_score_model(model_dir: str = 'sc_score_model/',
                   sc_score_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    sc_score_kwargs = sc_score_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier
    # CircularFingerprint
    model = ScScoreModel(**sc_score_kwargs)
    model.model.mode = 'classification'
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def chem_ception_model(model_dir: str = 'chem_ception_model/', chem_ception_kwargs: dict = None,
                       deepchem_kwargs: dict = None) -> DeepChemModel:
    chem_ception_kwargs = chem_ception_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # SmilesToImage
    model = ChemCeption(**chem_ception_kwargs)
    custom_objects = {'Stem': Stem, 'InceptionResnetA': InceptionResnetA, 'ReductionA': ReductionA,
                      'InceptionResnetB': InceptionResnetB, 'ReductionB': ReductionB,
                      'InceptionResnetC': InceptionResnetC}
    return DeepChemModel(model=model, model_dir=model_dir, custom_objects=custom_objects, **deepchem_kwargs)


def dag_model(model_dir: str = 'dag_model/', dag_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    dag_kwargs = dag_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # ConvMolFeaturizer
    model = DAGModel(**dag_kwargs)
    custom_objects = {'DAGLayer': DAGLayer, 'DAGGather': DAGGather}
    return DeepChemModel(model=model, model_dir=model_dir, custom_objects=custom_objects, **deepchem_kwargs)


def graph_conv_model(model_dir: str = 'graph_conv_model/', graph_conv_kwargs: dict = None,
                     deepchem_kwargs: dict = None) -> DeepChemModel:
    graph_conv_kwargs = graph_conv_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # ConvMolFeaturizer
    model = GraphConvModel(**graph_conv_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def smiles_to_vec_model(model_dir: str = 'smiles_to_vec_model/', smiles_to_vec_kwargs: dict = None,
                        deepchem_kwargs: dict = None) -> DeepChemModel:
    smiles_to_vec_kwargs = smiles_to_vec_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # SmilesToSeq
    model = Smiles2Vec(**smiles_to_vec_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def text_cnn_model(model_dir: str = 'text_cnn_model/',
                   text_cnn_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    text_cnn_kwargs = text_cnn_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    model = TextCNNModel(**text_cnn_kwargs)
    custom_objects = {"DTNNEmbedding": DTNNEmbedding, "Highway": Highway}
    return DeepChemModel(model=model, model_dir=model_dir, custom_objects=custom_objects, **deepchem_kwargs)


def weave_model(model_dir: str = 'weave_model/',
                weave_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    weave_kwargs = weave_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # WeaveFeaturizer
    model = WeaveModel(**weave_kwargs)
    custom_objects = {'WeaveLayer': WeaveLayer, 'TruncatedNormal': TruncatedNormal, 'WeaveGather': WeaveGather}
    return DeepChemModel(model=model, model_dir=model_dir, custom_objects=custom_objects, **deepchem_kwargs)


def dtnn_model(model_dir: str = 'dtnn_model/', dtnn_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    dtnn_kwargs = dtnn_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Regressor
    # CoulombMatrix
    model = DTNNModel(**dtnn_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def mat_model(model_dir: str = 'mat_model/', mat_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    mat_kwargs = mat_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Regressor
    # MATFeaturizer
    model = MATModel(**mat_kwargs)
    model.model.mode = 'regression'
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def multitask_regressor_model(model_dir: str = 'multitask_regressor_model/', multitask_regressor_kwargs: dict = None,
                              deepchem_kwargs: dict = None) -> DeepChemModel:
    multitask_regressor_kwargs = multitask_regressor_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Regressor
    # 1D Descriptors
    model = MultitaskRegressor(**multitask_regressor_kwargs)
    model.mode = 'regression'
    model.model.mode = 'regression'
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)
