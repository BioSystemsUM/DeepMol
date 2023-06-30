from deepchem.models import GATModel, GCNModel, AttentiveFPModel, PagtnModel, MPNNModel, MEGNetModel, DMPNNModel, CNN, \
    MultitaskClassifier, MultitaskIRVClassifier, MultitaskRegressor, ProgressiveMultitaskClassifier, \
    ProgressiveMultitaskRegressor, RobustMultitaskClassifier, RobustMultitaskRegressor, ScScoreModel, \
    ChemCeption, DAGModel, GraphConvModel, Smiles2Vec, TextCNNModel, DTNNModel, WeaveModel
from deepchem.models.chemnet_layers import Stem, InceptionResnetA, ReductionA, InceptionResnetB, ReductionB, \
    InceptionResnetC
from deepchem.models.layers import DTNNEmbedding, Highway, Stack, DAGLayer, DAGGather, WeaveLayer, WeaveGather
from deepchem.models.torch_models import MATModel
from keras.initializers.initializers import TruncatedNormal

from deepmol.models import DeepChemModel


def gat_model(model_dir: str = 'gat_model/', gat_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for GATModel from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#gatmodel
        - Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio.
        “Graph Attention Networks.” ICLR 2018.

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    gat_kwargs: dict
        Keyword arguments for GATModel.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped GATModel as DeepChemModel.
    """
    gat_kwargs = gat_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    model = GATModel(**gat_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def gcn_model(model_dir: str = 'gcn_model/', gcn_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for GCNModel from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#gcnmodel
        - Thomas N. Kipf and Max Welling. “Semi-Supervised Classification with Graph Convolutional Networks.” ICLR 2017.

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    gcn_kwargs: dict
        Keyword arguments for GCNModel.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped GCNModel as DeepChemModel.
    """
    gcn_kwargs = gcn_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    model = GCNModel(**gcn_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def attentivefp_model(model_dir: str = 'attentivefp_model/', attentivefp_kwargs: dict = None,
                      deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for AttentiveFPModel from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#attentivefpmodel
        - Zhaoping Xiong, Dingyan Wang, Xiaohong Liu, Feisheng Zhong, Xiaozhe Wan, Xutong Li, Zhaojun Li, Xiaomin Luo,
        Kaixian Chen, Hualiang Jiang, and Mingyue Zheng. “Pushing the Boundaries of Molecular Representation for Drug
        Discovery with the Graph Attention Mechanism.” Journal of Medicinal Chemistry. 2020, 63, 16, 8749–8760.

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    attentivefp_kwargs: dict
        Keyword arguments for AttentiveFPModel.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped AttentiveFPModel as DeepChemModel.
    """
    attentivefp_kwargs = attentivefp_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    model = AttentiveFPModel(**attentivefp_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def pagtn_model(model_dir: str = 'pagtn_model/', patgn_kwargs: dict = None,
                deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for PagtnModel from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pagtnmodel
        - Benson Chen, Regina Barzilay, Tommi Jaakkola. “Path-Augmented Graph Transformer Network.” arXiv:1905.12712

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    patgn_kwargs: dict
        Keyword arguments for PagtnModel.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped PagtnModel as DeepChemModel.
    """
    patgn_kwargs = patgn_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # PagtnMolGraphFeaturizer MolGraphConvFeaturizer
    model = PagtnModel(**patgn_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def mpnn_model(model_dir: str = 'mpnn_model/', mpnn_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for MPNNModel from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#id81
        - Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl. “Neural Message Passing
        for Quantum Chemistry.” ICML 2017.

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    mpnn_kwargs: dict
        Keyword arguments for MPNNModel.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped MPNNModel as DeepChemModel.
    """
    mpnn_kwargs = mpnn_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # MolGraphConvFeaturizer
    model = MPNNModel(**mpnn_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def megnet_model(model_dir: str = 'megnet_model/', megnet_kwargs: dict = None,
                 deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for MEGNetModel from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#megnetmodel
        - Chen, Chi, et al. “Graph networks as a universal machine learning framework for molecules and crystals.”
        Chemistry of Materials 31.9 (2019): 3564-3572.
        - Battaglia, Peter W., et al. “Relational inductive biases, deep learning, and graph networks.” arXiv preprint
        arXiv:1806.01261 (2018).

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    megnet_kwargs: dict
        Keyword arguments for MEGNetModel.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped MEGNetModel as DeepChemModel.
    """
    megnet_kwargs = megnet_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    model = MEGNetModel(**megnet_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def dmpnn_model(model_dir: str = 'dmpnn_model/', dmpnn_kwargs: dict = None,
                deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for DMPNNModel from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#dmpnnmodel
        - Analyzing Learned Molecular Representations for Property Prediction https://arxiv.org/pdf/1904.01561.pdf

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    dmpnn_kwargs: dict
        Keyword arguments for DMPNNModel.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped DMPNNModel as DeepChemModel.
    """
    dmpnn_kwargs = dmpnn_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # DMPNNFeaturizer
    model = DMPNNModel(**dmpnn_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def cnn_model(model_dir: str = 'cnn_model/', cnn_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for CNN from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#cnn

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    cnn_kwargs: dict
        Keyword arguments for CNN.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped CNN as DeepChemModel.
    """
    cnn_kwargs = cnn_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    model = CNN(**cnn_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def multitask_classifier_model(model_dir: str = 'multitask_classifier_model/', multitask_classifier_kwargs: dict = None,
                               deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for MultitaskClassifier from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#multitaskclassifier

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    multitask_classifier_kwargs: dict
        Keyword arguments for MultitaskClassifier.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped MultitaskClassifier as DeepChemModel.
    """
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
    """
    Deepmol wrapper for MultitaskIRVClassifier from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#tensorflowmultitaskirvclassifier

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    multitask_irv_classifier_kwargs: dict
        Keyword arguments for MultitaskIRVClassifier.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped MultitaskIRVClassifier as DeepChemModel.
    """
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
    """
    Deepmol wrapper for ProgressiveMultitaskClassifier from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#tensorflowmultitaskirvclassifier
        - Progressive Networks: https://arxiv.org/pdf/1606.04671v3.pdf

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    progressive_multitask_classifier_kwargs: dict
        Keyword arguments for ProgressiveMultitaskClassifier.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped ProgressiveMultitaskClassifier as DeepChemModel.
    """
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
    """
    Deepmol wrapper for ProgressiveMultitaskRegressor from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#progressivemultitaskregressor
        - Rusu, Andrei A., et al. “Progressive neural networks.” arXiv preprint arXiv:1606.04671 (2016).

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    progressive_multitask_regressor_kwargs: dict
        Keyword arguments for ProgressiveMultitaskRegressor.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped ProgressiveMultitaskRegressor as DeepChemModel.
    """
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
    """
    Deepmol wrapper for RobustMultitaskClassifier from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#robustmultitaskclassifier
        - Ramsundar, Bharath, et al. “Is multitask deep learning practical for pharma?.” Journal of chemical information
        and modeling 57.8 (2017): 2068-2076.

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    robust_multitask_classifier_kwargs: dict
        Keyword arguments for RobustMultitaskClassifier.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped RobustMultitaskClassifier as DeepChemModel.
    """
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
    """
    Deepmol wrapper for RobustMultitaskRegressor from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#robustmultitaskregressor
        - Ramsundar, Bharath, et al. “Is multitask deep learning practical for pharma?.” Journal of chemical information
        and modeling 57.8 (2017): 2068-2076.

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    robust_multitask_regressor_kwargs: dict
        Keyword arguments for RobustMultitaskRegressor.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped RobustMultitaskRegressor as DeepChemModel.
    """
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
    """
    Deepmol wrapper for ScScoreModel from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#scscoremodel
        - https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00622

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    sc_score_kwargs: dict
        Keyword arguments for ScScoreModel.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped ScScoreModel as DeepChemModel.
    """
    sc_score_kwargs = sc_score_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier
    # CircularFingerprint
    model = ScScoreModel(**sc_score_kwargs)
    model.model.mode = 'classification'
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def chem_ception_model(model_dir: str = 'chem_ception_model/', chem_ception_kwargs: dict = None,
                       deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for ChemCeption from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#chemception
        - Goh et al., “Chemception: A Deep Neural Network with Minimal Chemistry Knowledge Matches the Performance of
        Expert-developed QSAR/QSPR Models” (https://arxiv.org/pdf/1706.06689.pdf)

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    chem_ception_kwargs: dict
        Keyword arguments for ChemCeption.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped ChemCeption as DeepChemModel.
    """
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
    """
    Deepmol wrapper for DAGModel from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#dagmodel
        - Lusci, Alessandro, Gianluca Pollastri, and Pierre Baldi. “Deep architectures and deep learning in
        chemoinformatics: the prediction of aqueous solubility for drug-like molecules.” Journal of chemical information
        and modeling 53.7 (2013): 1563-1575.

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    dag_kwargs: dict
        Keyword arguments for DAGModel.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped DAGModel as DeepChemModel.
    """
    dag_kwargs = dag_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # ConvMolFeaturizer
    model = DAGModel(**dag_kwargs)
    custom_objects = {'DAGLayer': DAGLayer, 'DAGGather': DAGGather}
    return DeepChemModel(model=model, model_dir=model_dir, custom_objects=custom_objects, **deepchem_kwargs)


def graph_conv_model(model_dir: str = 'graph_conv_model/', graph_conv_kwargs: dict = None,
                     deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for GraphConvModel from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#graphconvmodel
        - Duvenaud, David K., et al. “Convolutional networks on graphs for learning molecular fingerprints.” Advances in
        neural information processing systems. 2015.

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    graph_conv_kwargs: dict
        Keyword arguments for GraphConvModel.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped GraphConvModel as DeepChemModel.
    """
    graph_conv_kwargs = graph_conv_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # ConvMolFeaturizer
    model = GraphConvModel(**graph_conv_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def smiles_to_vec_model(model_dir: str = 'smiles_to_vec_model/', smiles_to_vec_kwargs: dict = None,
                        deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for Smiles2Vec from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#smiles2vec
        - Goh et al., “SMILES2vec: An Interpretable General-Purpose Deep Neural Network for Predicting Chemical
        Properties” (https://arxiv.org/pdf/1712.02034.pdf).

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    smiles_to_vec_kwargs: dict
        Keyword arguments for Smiles2Vec.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped Smiles2Vec as DeepChemModel.
    """
    smiles_to_vec_kwargs = smiles_to_vec_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # SmilesToSeq
    model = Smiles2Vec(**smiles_to_vec_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def text_cnn_model(model_dir: str = 'text_cnn_model/',
                   text_cnn_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for TextCNNModel from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#textcnnmodel
        - Guimaraes, Gabriel Lima, et al. “Objective-reinforced generative adversarial networks (ORGAN) for sequence
        generation models.” arXiv preprint arXiv:1705.10843 (2017).
        - Kim, Yoon. “Convolutional neural networks for sentence classification.” arXiv preprint arXiv:1408.5882 (2014).

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    text_cnn_kwargs: dict
        Keyword arguments for TextCNNModel.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped TextCNNModel as DeepChemModel.
    """
    text_cnn_kwargs = text_cnn_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    model = TextCNNModel(**text_cnn_kwargs)
    custom_objects = {"DTNNEmbedding": DTNNEmbedding, "Highway": Highway}
    return DeepChemModel(model=model, model_dir=model_dir, custom_objects=custom_objects, **deepchem_kwargs)


def weave_model(model_dir: str = 'weave_model/',
                weave_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for WeaveModel from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#weavemodel
        - Kearnes, Steven, et al. “Molecular graph convolutions: moving beyond fingerprints.” Journal of computer-aided
        molecular design 30.8 (2016): 595-608.

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    weave_kwargs: dict
        Keyword arguments for WeaveModel.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped WeaveModel as DeepChemModel.
    """
    weave_kwargs = weave_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Classifier/ Regressor
    # WeaveFeaturizer
    model = WeaveModel(**weave_kwargs)
    custom_objects = {'WeaveLayer': WeaveLayer, 'TruncatedNormal': TruncatedNormal, 'WeaveGather': WeaveGather}
    return DeepChemModel(model=model, model_dir=model_dir, custom_objects=custom_objects, **deepchem_kwargs)


def dtnn_model(model_dir: str = 'dtnn_model/', dtnn_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for DTNNModel from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#dtnnmodel
        - Schütt, Kristof T., et al. “Quantum-chemical insights from deep tensor neural networks.” Nature communications
        8.1 (2017): 1-8.

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    dtnn_kwargs: dict
        Keyword arguments for DTNNModel.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped DTNNModel as DeepChemModel.
    """
    dtnn_kwargs = dtnn_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Regressor
    # CoulombMatrix
    model = DTNNModel(**dtnn_kwargs)
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def mat_model(model_dir: str = 'mat_model/', mat_kwargs: dict = None, deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for MATModel from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#matmodel
        - Lukasz Maziarka et al. “Molecule Attention Transformer” Graph Representation Learning workshop and Machine
        Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    mat_kwargs: dict
        Keyword arguments for MATModel.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped MATModel as DeepChemModel.
    """
    mat_kwargs = mat_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Regressor
    # MATFeaturizer
    model = MATModel(**mat_kwargs)
    model.model.mode = 'regression'
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)


def multitask_regressor_model(model_dir: str = 'multitask_regressor_model/', multitask_regressor_kwargs: dict = None,
                              deepchem_kwargs: dict = None) -> DeepChemModel:
    """
    Deepmol wrapper for MultitaskRegressor from DeepChem.
    References:
        - https://deepchem.readthedocs.io/en/latest/api_reference/models.html#multitaskregressor

    Parameters
    ----------
    model_dir: str
        Directory where the model will be saved.
    multitask_regressor_kwargs: dict
        Keyword arguments for MultitaskRegressor.
    deepchem_kwargs: dict
        Keyword arguments for DeepChemModel class.

    Returns
    -------
    DeepChemModel
        Wrapped MultitaskRegressor as DeepChemModel.
    """
    multitask_regressor_kwargs = multitask_regressor_kwargs or {}
    deepchem_kwargs = deepchem_kwargs or {}
    # Regressor
    # 1D Descriptors
    model = MultitaskRegressor(**multitask_regressor_kwargs)
    model.mode = 'regression'
    model.model.mode = 'regression'
    return DeepChemModel(model=model, model_dir=model_dir, **deepchem_kwargs)
