from models.Models import Model
from metrics.Metrics import Metric
from Dataset.Dataset import Dataset
from typing import Dict, Any, Optional, Tuple
import collections
import numpy as np
import os
import shutil
import tempfile
from functools import reduce
from operator import mul
import itertools
import random


def _convert_hyperparam_dict_to_filename(hyper_params: Dict[str, Any]) -> str:
    """Function that converts a dictionary of hyperparameters to a string that can be a filename.
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
            filename += "%s" % str(value)
    return filename


class HyperparamOpt(object):
    """Abstract superclass for hyperparameter search classes.
    """

    def __init__(self, model_builder: Model):
      """Initialize Hyperparameter Optimizer.
      Note this is an abstract constructor which should only be used by subclasses.

      Parameters
      ----------
      model_builder: constructor function.
        This parameter must be constructor function which returns an
        object which is an instance of `Models`. This function
        must accept two arguments, `model_params` of type `dict` and
        'model_dir', a string specifying a path to a model directory.
      """
      if self.__class__.__name__ == "HyperparamOpt":
          raise ValueError("HyperparamOpt is an abstract superclass and cannot be directly instantiated. "
                            "You probably want to instantiate a concrete subclass instead.")
      self.model_builder = model_builder

    def hyperparam_search(self,
                          params_dict: Dict[str, Any],
                          train_dataset: Dataset,
                          valid_dataset: Dataset,
                          metric: Metric,
                          use_max: bool = True,
                          logdir: Optional[str] = None,
                          **kwargs) -> Tuple[Model, Dict[str, Any], Dict[str, float]]:
      """Conduct Hyperparameter search.
      This method defines the common API shared by all hyperparameter
      optimization subclasses. Different classes will implement
      different search methods but they must all follow this common API.

      Parameters
      ----------
      params_dict: Dict
        Dictionary mapping strings to values. Note that the
        precise semantics of `params_dict` will change depending on the
        optimizer that you're using.
      train_dataset: Dataset
        dataset used for training
      valid_dataset: Dataset
        dataset used for validation(optimization on valid scores)
      metric: Metric
        metric used for evaluation
      use_max: bool, optional
        If True, return the model with the highest score.
      logdir: str, optional
        The directory in which to store created models. If not set, will
        use a temporary directory.
      Returns
      -------
      Tuple[`best_model`, `best_hyperparams`, `all_scores`]
      """
      raise NotImplementedError


class GridHyperparamOpt(HyperparamOpt):
    """
    Provides simple grid hyperparameter search capabilities.
    This class performs a grid hyperparameter search over the specified
    hyperparameter space.
    """

    def hyperparam_search(self,
                          params_dict: Dict,
                          train_dataset: Dataset,
                          valid_dataset: Dataset,
                          metric: Metric,
                          n_iter_search: int = 15,
                          use_max: bool = True,
                          logdir: Optional[str] = None,
                          **kwargs):
        """Perform hyperparams search according to params_dict.
        Each key to hyperparams_dict is a model_param. The values should
        be a list of potential values for that hyperparam.
        Parameters
        ----------
        params_dict: Dict
          Maps hyperparameter names (strings) to lists of possible
          parameter values.
        train_dataset: Dataset
          dataset used for training
        valid_dataset: Dataset
          dataset used for validation(optimization on valid scores)
        metric: Metric
          metric used for evaluation
        n_iter_search: int or None, optional
            Number of random combinations of parameters to test, if None performs complete grid search
        use_max: bool, optional
          If True, return the model with the highest score.
        logdir: str, optional
          The directory in which to store created models. If not set, will
          use a temporary directory.
        Returns
        -------
        Tuple[`best_model`, `best_hyperparams`, `all_scores`]
        """

        hyperparams = params_dict.keys()
        hyperparam_vals = params_dict.values()
        for hyperparam_list in params_dict.values():
            assert isinstance(hyperparam_list, collections.Iterable)

        number_combinations = reduce(mul, [len(vals) for vals in hyperparam_vals])

        if use_max:
            best_validation_score = -np.inf
        else:
            best_validation_score = np.inf

        # To make sure that the number of iterations is lower or equal to the number of max hypaparameter combinations
        len_params = sum(1 for x in itertools.product(*params_dict.values()))
        if n_iter_search is None or len_params < n_iter_search:
                n_iter_search = len_params
        print(len_params, n_iter_search)
        random_inds = random.sample(range(0, len_params), k=n_iter_search)
        print('??????????????????????', random_inds)
        best_hyperparams = None
        best_model, best_model_dir = None, None
        all_scores = {}
        j = 0
        print("Fitting %d random models from a space of %d possible models." % (len(random_inds), number_combinations))
        for ind, hyperparameter_tuple in enumerate(itertools.product(*hyperparam_vals)):
            if ind in random_inds:
                j+=1
                model_params = {}
                print("Fitting model %d/%d" % (j, len(random_inds)))
                hyper_params = dict(zip(hyperparams, hyperparameter_tuple))
                for hyperparam, hyperparam_val in zip(hyperparams, hyperparameter_tuple):
                    model_params[hyperparam] = hyperparam_val
                print("hyperparameters: %s" % str(model_params))

                if logdir is not None:
                    model_dir = os.path.join(logdir, str(ind))
                    print("model_dir is %s" % model_dir)
                    try:
                        os.makedirs(model_dir)
                    except OSError:
                        if not os.path.isdir(model_dir):
                            print("Error creating model_dir, using tempfile directory")
                            model_dir = tempfile.mkdtemp()
                else:
                    model_dir = tempfile.mkdtemp()

                model_params['model_dir'] = model_dir
                model = self.model_builder(**model_params)
                model.fit(train_dataset)
                try:
                    model.save()
                except NotImplementedError:
                    pass

                multitask_scores = model.evaluate(valid_dataset, [metric])
                valid_score = multitask_scores[metric.name]
                hp_str = _convert_hyperparam_dict_to_filename(hyper_params)
                all_scores[hp_str] = valid_score

                if (use_max and valid_score >= best_validation_score) or (
                        not use_max and valid_score <= best_validation_score):
                    best_validation_score = valid_score
                    best_hyperparams = hyperparameter_tuple
                    if best_model_dir is not None:
                        shutil.rmtree(best_model_dir)
                    best_model_dir = model_dir
                    best_model = model
                else:
                    shutil.rmtree(model_dir)

                print("Model %d/%d, Metric %s, Validation set %s: %f" % (
                j, len(random_inds), metric.name, j, valid_score))
                print("\tbest_validation_score so far: %f" % best_validation_score)

        if best_model is None:
            print("No models trained correctly.")
            # arbitrarily return last model
            best_model, best_hyperparams = model, hyperparameter_tuple
            return best_model, best_hyperparams, all_scores

        multitask_scores = best_model.evaluate(train_dataset, [metric])
        train_score = multitask_scores[metric.name]
        print("Best hyperparameters: %s" % str(best_hyperparams))
        print("train_score: %f" % train_score)
        print("validation_score: %f" % best_validation_score)
        return best_model, best_hyperparams, all_scores
