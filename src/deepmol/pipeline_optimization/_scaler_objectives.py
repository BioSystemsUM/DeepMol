import os

from deepmol.base import PassThroughTransformer, Transformer
from deepmol.scalers import *


def standard_scaler_step(trial) -> Transformer:
    """
    Get a StandardScaler object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Transformer
        The StandardScaler object step.
    """
    with_mean = trial.suggest_categorical("with_mean", [True, False])
    with_std = trial.suggest_categorical("with_std", [True, False])
    return StandardScaler(with_mean=with_mean, with_std=with_std)


def min_max_scaler_step(trial) -> Transformer:
    """
    Get a MinMaxScaler object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Transformer
        The MinMaxScaler object step.
    """
    return MinMaxScaler()


def max_abs_scaler_step(trial) -> Transformer:
    """
    Get a MaxAbsScaler object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Transformer
        The MaxAbsScaler object step.
    """
    return MaxAbsScaler()


def robust_scaler_step(trial) -> Transformer:
    """
    Get a RobustScaler object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Transformer
        The RobustScaler object step.
    """
    with_centering = trial.suggest_categorical("with_centering", [True, False])
    with_scaling = trial.suggest_categorical("with_scaling", [True, False])
    quantile_range = trial.suggest_categorical("quantile_range",
                                               [str(cat) for cat in [(25.0, 75.0), (10.0, 90.0), (5.0, 95.0)]])
    return RobustScaler(with_centering=with_centering, with_scaling=with_scaling, quantile_range=eval(quantile_range))


def normalizer_step(trial) -> Transformer:
    """
    Get a Normalizer object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Transformer
        The Normalizer object step.
    """
    norm = trial.suggest_categorical("norm", ["l1", "l2", "max"])
    return Normalizer(norm=norm)


def binarizer_step(trial) -> Transformer:
    """
    Get a Binarizer object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Transformer
        The Binarizer object step.
    """
    return Binarizer()


def quantile_transformer_step(trial) -> Transformer:
    """
    Get a QuantileTransformer object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Transformer
        The QuantileTransformer object step.
    """
    n_quantiles = trial.suggest_int("n_quantiles", 10, 1000)
    output_distribution = trial.suggest_categorical("output_distribution", ["uniform", "normal"])
    return QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution)


def power_transformer_step(trial) -> Transformer:
    """
    Get a PowerTransformer object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Transformer
        The PowerTransformer object step.
    """
    method = trial.suggest_categorical("method", ["yeo-johnson", "box-cox"])
    standardize = trial.suggest_categorical("standardize", [True, False])
    return PowerTransformer(method=method, standardize=standardize)


def pass_through_transformer_step(trial) -> Transformer:
    """
    Get a PassThroughTransformer object for the Optuna optimization.
    This scaler does not scale the input data.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Transformer
        The PassThroughTransformer object step.
    """
    return PassThroughTransformer()


# TODO: PolynomialFeatures only woks with square matrices
# TODO: KernelCenter only woks with square matrices
_SCALERS = {'standard_scaler': standard_scaler_step, 'min_max_scaler': min_max_scaler_step,
            'max_abs_scaler': max_abs_scaler_step, 'robust_scaler': robust_scaler_step, 'normalizer': normalizer_step,
            'binarizer': binarizer_step, 'quantile_transformer': quantile_transformer_step,
            'power_transformer': power_transformer_step, 'pass_through_transformer': pass_through_transformer_step}


def _get_scaler(trial) -> Transformer:
    """
    Get a scaler object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Transformer
        The scaler object step.
    """
    scaler = trial.suggest_categorical("scaler", list(_SCALERS.keys()))
    return _SCALERS[scaler](trial)
