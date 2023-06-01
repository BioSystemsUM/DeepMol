from optuna import Trial

from deepmol.base import PassThroughTransformer, Transformer
from deepmol.feature_selection import KbestFS, LowVarianceFS, PercentilFS, RFECVFS, SelectFromModelFS, BorutaAlgorithm

_FEATURE_SELECTORS = {"k_best": KbestFS, "low_variance_fs": LowVarianceFS, "percentil_fs": PercentilFS,
                      "rfecvfs": RFECVFS, "select_from_model_fs": SelectFromModelFS,
                      "boruta_algorithm": BorutaAlgorithm, "pass_through_transformer": PassThroughTransformer}


def _get_feature_selector(trial: Trial) -> Transformer:
    feature_selector = trial.suggest_categorical("feature_selector", list(_FEATURE_SELECTORS.keys()))
    # TODO: add parameters for some feature selectors
    return _FEATURE_SELECTORS[feature_selector]()
