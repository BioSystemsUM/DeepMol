from deepmol.base import PassThroughTransformer, Transformer
from deepmol.standardizer import *
from deepmol.standardizer._utils import simple_standardisation, heavy_standardisation


_STANDARDIZERS = {"basic_standardizer": BasicStandardizer,
                  "custom_standardizer": CustomStandardizer,
                  "chembl_standardizer": ChEMBLStandardizer,
                  'pass_through_standardizer': PassThroughTransformer}


def _get_standardizer(trial) -> Transformer:
    """
    Get a Standardizer object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Transformer
        The Standardizer object step.
    """
    standardizer = trial.suggest_categorical("standardizer", list(_STANDARDIZERS.keys()))
    if standardizer == "custom_standardizer":
        choice = trial.suggest_categorical("standardization_type", ["simple_standardisation", "heavy_standardisation"])
        if choice == "simple_standardisation":
            params = simple_standardisation
        else:
            params = heavy_standardisation
        return CustomStandardizer(params)
    else:
        return _STANDARDIZERS[standardizer]()
