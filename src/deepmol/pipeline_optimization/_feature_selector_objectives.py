from optuna import Trial
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_regression

from deepmol.base import PassThroughTransformer, Transformer
from deepmol.feature_selection import *


def k_best_fs(trial: Trial, task_type: str):
    """
    Optuna objective function for KbestFS.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    task_type : str
        The type of the task. Either 'classification' or 'regression'.

    Returns
    -------
    KbestFS
        A KbestFS object.
    """
    if task_type == 'classification':
        return KbestFS(k=trial.suggest_int("k", 1, 25))
    else:
        return KbestFS(k=trial.suggest_int("k", 1, 25), score_func=f_regression)


def low_variance_fs(trial: Trial, task_type: str):
    """
    Optuna objective function for LowVarianceFS.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    task_type : str
        The type of the task. Either 'classification' or 'regression'.

    Returns
    -------
    LowVarianceFS
        A LowVarianceFS object.
    """
    return LowVarianceFS(threshold=trial.suggest_float("threshold", 0, 0.15))


def percentil_fs(trial: Trial, task_type: str):
    """
    Optuna objective function for PercentilFS.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    task_type : str
        The type of the task. Either 'classification' or 'regression'.

    Returns
    -------
    PercentilFS
        A PercentilFS object.
    """
    if task_type == 'classification':
        return PercentilFS(percentil=trial.suggest_int("percentile", 0, 100))
    else:
        return PercentilFS(percentil=trial.suggest_int("percentile", 0, 100), score_func=f_regression)


def rfecv_fs(trial: Trial, task_type: str):
    """
    Optuna objective function for RFECVFS.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    task_type : str
        The type of the task. Either 'classification' or 'regression'.

    Returns
    -------
    RFECVFS
        A RFECVFS object.
    """
    if task_type == 'classification':
        estimator = RandomForestClassifier(n_estimators=trial.suggest_int("n_estimators", 10, 1000))
        return RFECVFS(estimator=estimator, step=trial.suggest_loguniform("step", 0.01, 1))
    else:
        estimator = RandomForestRegressor(n_estimators=trial.suggest_int("n_estimators", 10, 1000))
        return RFECVFS(estimator=estimator, step=trial.suggest_loguniform("step", 0.01, 1))


def select_from_model_fs(trial: Trial, task_type: str):
    """
    Optuna objective function for SelectFromModelFS.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    task_type : str
        The type of the task. Either 'classification' or 'regression'.

    Returns
    -------
    SelectFromModelFS
        A SelectFromModelFS object.
    """
    if task_type == 'classification':
        estimator = RandomForestClassifier(n_estimators=trial.suggest_int("n_estimators", 10, 1000))
        return SelectFromModelFS(estimator=estimator, threshold="median")
    else:
        estimator = RandomForestRegressor(n_estimators=trial.suggest_int("n_estimators", 10, 1000))
        return SelectFromModelFS(estimator=estimator, threshold="median")


def boruta_algorithm_fs(trial: Trial, task_type: str):
    """
    Optuna objective function for BorutaAlgorithm.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    task_type : str
        The type of the task. Either 'classification' or 'regression'.

    Returns
    -------
    BorutaAlgorithm
        A BorutaAlgorithm object.
    """
    support_weak = trial.suggest_categorical("support_weak", [True, False])
    if task_type == 'classification':
        return BorutaAlgorithm(n_estimators="auto", support_weak=support_weak)
    else:
        return BorutaAlgorithm(task='regression', n_estimators="auto", support_weak=support_weak)


def pass_through_transformer(trial: Trial, task_type: str):
    """
    Optuna objective function for PassThroughTransformer.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    task_type : str
        The type of the task. Either 'classification' or 'regression'.

    Returns
    -------
    PassThroughTransformer
        A PassThroughTransformer object.
    """
    return PassThroughTransformer()


_FEATURE_SELECTORS = {"k_best": k_best_fs, "low_variance_fs": low_variance_fs, "percentil_fs": percentil_fs,
                      "select_from_model_fs": select_from_model_fs,
                      "boruta_algorithm": boruta_algorithm_fs, "pass_through_transformer": pass_through_transformer}


def _get_feature_selector(trial: Trial, task_type: str, multitask: bool) -> Transformer:
    """
    Get a feature selector object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    task_type : str
        The type of the task. Either 'classification' or 'regression'.
    multitask : bool
        Whether the task is multitask or not.

    Returns
    -------
    Transformer
        A feature selector object.
    """
    if multitask:
        feature_selector = trial.suggest_categorical("feature_selector", ["pass_through_transformer",
                                                                          "low_variance_fs"])
    else:
        feature_selector = trial.suggest_categorical("feature_selector", list(_FEATURE_SELECTORS.keys()))
    return _FEATURE_SELECTORS[feature_selector](trial, task_type)
