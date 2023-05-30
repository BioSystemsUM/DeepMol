from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.pipeline import Pipeline

_STANDARDIZERS = ["deepmol.standardizer.BasicStandardizer",
                  "deepmol.standardizer.CustomStandardizer",
                  "deepmol.standardizer.ChEMBLStandardizer",
                  "deepmol.base.PassThroughTransformer"]

# TODO: MixedFeaturizer, DeepChemFeaturizers, One-Hot-Encoder, ...
_FEATURIZERS = ["deepmol.compound_featurization.All3DDescriptors",
                "deepmol.compound_featurization.TwoDimensionDescriptors",
                "deepmol.compound_featurization.MorganFingerprint",
                "deepmol.compound_featurization.AtomPairFingerprint",
                "deepmol.compound_featurization.LayeredFingerprint",
                "deepmol.compound_featurization.RDKFingerprint",
                "deepmol.compound_featurization.MACCSkeysFingerprint",
                "deepmol.compound_featurization.Mol2Vec"]

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
_MODELS = []

# TODO: How to deal with incompatible steps? (e.g. some deepchem featurizers only work with some deepchem models)


def _get_preset(train_dataset: Dataset, test_dataset: Dataset, metric: Metric, preset: str) -> callable:
    pass


def heavy_preset(trial, train_dataset, test_dataset, metric) -> callable:
    # standardizer
    standardizer = trial.suggest_categorical("standardizer", _STANDARDIZERS)
    # featurizer
    # scaler
    # feature_selector
    # model
    steps = []
    pipeline = Pipeline(steps=steps)
    pipeline.fit_transform(train_dataset)
    score = pipeline.evaluate(test_dataset, metric)[0][metric.name]
    return score
