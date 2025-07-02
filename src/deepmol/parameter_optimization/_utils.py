import inspect
from typing import Dict, Any, Union, Callable

import sklearn
from sklearn.metrics import make_scorer
import sklearn.metrics

from deepmol.loggers.logger import Logger
from deepmol.metrics import Metric


def _convert_hyperparam_dict_to_filename(hyper_params: Dict[str, Any]) -> str:
    """
    Function that converts a dictionary of hyperparameters to a string that can be a filename.

    Parameters
    ----------
    hyper_params: Dict
      Maps string of hyperparameter name to int/float/string/list etc.

    Returns
    -------
    filename: str
      A filename of form "_key1_value1_value2_..._key2..."
    """
    filename = ""
    keys = sorted(hyper_params.keys())
    for key in keys:
        filename += "_%s" % str(key)
        value = hyper_params[key]
        if isinstance(value, int):
            filename += "_%s" % str(value)
        elif isinstance(value, float):
            filename += "_%f" % value
        else:
            filename += "_%s" % str(value)
    return filename


def validate_metrics(metric: Union[str, Metric, Callable]) -> Union[str, Callable]:
    """
    Validate single and multi metrics.

    Parameters
    ----------
    metric: Union[Callable, str, Metric]
        The metrics to validate.

    Returns
    -------
    metric: Union[str, Callable]
        Validated metric.
    """

    logger = Logger()
    sklearn_functions = [
        name for name, _ in inspect.getmembers(sklearn.metrics, inspect.isfunction)
        ]
    
    if isinstance(metric, Metric):
        logger.info(f'Using {metric.name} as scoring function.')
        return make_scorer(metric.metric)
    elif isinstance(metric, Callable):
        metric_str = metric.__name__
        if metric_str in sklearn_functions:
            logger.info(f'Using {metric.__name__} as scoring function.')
            return metric
        
        metric = Metric(metric)
        logger.info(f'Using {metric.name} as scoring function.')
        return make_scorer(metric)
    elif isinstance(metric, str) and metric in sklearn_functions:
        logger.info(f'Using {metric} as scoring function.')
        return metric
    else:
        raise ValueError(f'{metric}, is not a valid scoring function. '
                         'Use sorted(sklearn.metrics.SCORERS.keys()) to get valid options.')
