from deepmol.base import PassThroughTransformer
from deepmol.compound_featurization import All3DDescriptors, TwoDimensionDescriptors, MorganFingerprint, \
    AtomPairFingerprint, LayeredFingerprint, RDKFingerprint, MACCSkeysFingerprint, Mol2Vec, MixedFeaturizer
from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.pipeline import Pipeline
from deepmol.standardizer import BasicStandardizer, ChEMBLStandardizer, CustomStandardizer
from deepmol.standardizer._utils import simple_standardisation, heavy_standardisation

_STANDARDIZERS = {"basic_standardizer": BasicStandardizer(),
                  "custom_standardizer": CustomStandardizer(),
                  "chembl_standardizer": ChEMBLStandardizer(),
                  'pass_through_standardizer': PassThroughTransformer()}


def _get_standardizer(trial, standardizer):
    if standardizer == "custom_standardizer":
        choice = trial.suggest_categorical("choice", ["simple_standardisation", "heavy_standardisation"])
        if choice == "simple_standardisation":
            params = simple_standardisation
        else:
            params = heavy_standardisation
        return CustomStandardizer(params)
    else:
        return _STANDARDIZERS[standardizer]


# TODO: MixedFeaturizer, DeepChemFeaturizers, One-Hot-Encoder, ...
_BASIC_FEATURIZERS = {"all_3d_descriptors": All3DDescriptors(),
                      "two_dimension_descriptors": TwoDimensionDescriptors(),
                      "morgan_fingerprint": MorganFingerprint(),
                      "atom_pair_fingerprint": AtomPairFingerprint(),
                      "layered_fingerprint": LayeredFingerprint(),
                      "rdk_fingerprint": RDKFingerprint(),
                      "maccs_keys_fingerprint": MACCSkeysFingerprint(),
                      "mol2vec": Mol2Vec()}

#_ONE_HOT_ENCODER = SmilesOneHotEncoder()

_MIXED_FEATURIZER = "mixed_featurizer"


def _get_featurizer(trial, featurizer):
    if featurizer == "morgan_fingerprint":
        radius = trial.suggest_int("radius", 2, 6, step=2)
        n_bits = trial.suggest_int("n_bits", 1024, 2048, step=1024)
        chiral = trial.suggest_categorical("chiral", [True, False])
        bonds = trial.suggest_categorical("bonds", [True, False])
        features = trial.suggest_categorical("features", [True, False])
        return MorganFingerprint(radius=radius, size=n_bits, chiral=chiral, bonds=bonds, features=features)
    elif featurizer == "atom_pair_fingerprint":
        n_bits = trial.suggest_int("n_bits", 1024, 2048, step=1024)
        min_len = trial.suggest_int("min_len", 1, 3)
        max_len = trial.suggest_int("max_len", 30, 50, step=10)
        n_bist_per_entry = trial.suggest_int("n_bits_per_entry", 2, 8, step=2)
        include_chirality = trial.suggest_categorical("include_chirality", [True, False])
        use_2d = trial.suggest_categorical("use_2d", [True, False])
        return AtomPairFingerprint(size=n_bits, min_length=min_len, max_length=max_len,
                                   n_bits_per_entry=n_bist_per_entry, include_chirality=include_chirality,
                                   use_2d=use_2d)
    # TODO: add other featurizers with hyperparameters
    else:
        return _BASIC_FEATURIZERS[featurizer]


# TODO: PolynomialFeatures only woks with square matrices
_SCALERS = ["deepmol.scaler.StandardScaler",
            "deepmol.scaler.MinMaxScaler",
            "deepmol.scaler.MaxAbsScaler",
            "deepmol.scaler.RobustScaler",
            "deepmol.scaler.Normalizer",
            "deepmol.scaler.Binarizer",
            "deepmol.scaler.KernelCenterer",
            "deepmol.scaler.QuantileTransformer",
            "deepmol.scaler.PowerTransformer",
            "deepmol.base.PassThroughTransformer"]

_FEATURE_SELECTORS = ["deepmol.feature_selection.KBest",
                      "deepmol.feature_selection.LowVarianceFS",
                      "deepmol.feature_selection.PercentilFS",
                      "deepmol.feature_selection.RFECVFS",
                      "deepmol.feature_selection.SelectFromModelFS",
                      "deepmol.feature_selection.BorutaAlgorithm",
                      "deepmol.base.PassThroughTransformer"]

# TODO: add more models
_BASE_MODELS = []
_RNN_MODELS = []
_CNN_MODELS = []


# TODO: How to deal with incompatible steps? (e.g. some deepchem featurizers only work with some deepchem models)


def _get_preset(train_dataset: Dataset, test_dataset: Dataset, metric: Metric, preset: str) -> callable:
    pass


def heavy_objective(trial, train_dataset, test_dataset, metric) -> callable:
    # model
    model = trial.suggest_categorical("model", _BASE_MODELS)
    # standardizer
    standardizer = trial.suggest_categorical("standardizer", _STANDARDIZERS.keys())
    standardizer = _get_standardizer(trial, standardizer)
    # featurizer
    featurizer = trial.suggest_categorical("featurizer", list(_BASIC_FEATURIZERS.keys()) + [_MIXED_FEATURIZER])
    # scaler
    scaler = trial.suggest_categorical("scaler", _SCALERS)
    # feature_selector
    feature_selector = trial.suggest_categorical("feature_selector", _FEATURE_SELECTORS)
    steps = [('standardizer', standardizer), ('featurizer', featurizer), ('scaler', scaler),
             ('feature_selector', feature_selector), ('model', model)]
    pipeline = Pipeline(steps=steps)
    pipeline.fit_transform(train_dataset)
    score = pipeline.evaluate(test_dataset, metric)[0][metric.name]
    return score
