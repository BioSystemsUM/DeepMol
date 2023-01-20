"""Hyperparameter Optimization Class"""
import collections
import itertools
import os
import random
import shutil
import tempfile
from functools import reduce
from operator import mul
from typing import Dict, Any, Union, Tuple

import numpy as np
import sklearn
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV, GridSearchCV

from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.models import SklearnModel, KerasModel
from deepmol.models.models import Model
from deepmol.parameter_optimization.deepchem_hyperparameter_optimization import DeepchemRandomSearchCV, \
    DeepchemGridSearchCV


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
            filename += "%s" % str(value)
    return filename


def validate_metrics(metrics: Union[Dict, str, Metric]):
    """
    Validate single and multi metrics.

    Parameters
    ----------
    metrics: Union[Dict, str, Metric]
        The metrics to validate.

    Returns
    -------
    all_metrics: List
        The list of validated metrics.
    """
    if isinstance(metrics, dict):
        all_metrics = []
        for m in metrics.values():
            if m in sklearn.metrics.SCORERS.keys() or isinstance(m, sklearn.metrics._scorer._PredictScorer):
                all_metrics.append(m)
            else:
                print(m, ' is not a valid scoring function. Use sorted(sklearn.metrics.SCORERS.keys()) '
                         'to get valid options.')
    else:
        if metrics in sklearn.metrics.SCORERS.keys() or isinstance(metrics, sklearn.metrics._scorer._PredictScorer):
            all_metrics = metrics
        else:
            print('WARNING: ', metrics, ' is not a valid scoring function. '
                                        'Use sorted(sklearn.metrics.SCORERS.keys()) to get valid options.')

    if not metrics:
        metrics = 'accuracy'
        print('Using accuracy instead and ', metrics, ' on validation!\n \n')

    return metrics


class HyperparameterOptimizer(object):
    """
    Abstract superclass for hyperparameter search classes.
    """

    def __init__(self, model_builder: callable, mode: str = None):
        """
        Initialize Hyperparameter Optimizer.
        Note this is an abstract constructor which should only be used by subclasses.

        Parameters
        ----------
        model_builder: callable
            This parameter must be constructor function which returns an object which is an instance of `Models`.
            This function must accept two arguments, `model_params` of type `dict` and 'model_dir', a string specifying
            a path to a model directory.
        mode: str
            The mode of the model. Can be 'classification' or 'regression'.
        """
        if self.__class__.__name__ == "HyperparamOpt":
            raise ValueError("HyperparamOpt is an abstract superclass and cannot be directly instantiated. "
                             "You probably want to instantiate a concrete subclass instead.")
        self.model_builder = model_builder
        self.mode = mode

    def hyperparameter_search(self,
                              params_dict: Dict[str, Any],
                              train_dataset: Dataset,
                              valid_dataset: Dataset,
                              metric: Metric,
                              use_max: bool = True,
                              logdir: str = None,
                              **kwargs) -> Tuple[Model, Dict[str, Any], Dict[str, float]]:
        """
        Conduct Hyperparameter search.

        This method defines the common API shared by all hyperparameter optimization subclasses. Different classes will
        implement different search methods, but they must all follow this common API.

        Parameters
        ----------
        params_dict: Dict
            Dictionary mapping strings to values. Note that the precise semantics of `params_dict` will change depending
            on the optimizer that you're using.
        train_dataset: Dataset
            The training dataset.
        valid_dataset: Dataset
            The validation dataset.
        metric: Metric
            The metric to optimize.
        use_max: bool
            If True, return the model with the highest score.
        logdir: str
            The directory in which to store created models. If not set, will use a temporary directory.
        **kwargs: Dict
            Additional keyword arguments to pass to the model constructor.

        Returns
        -------
        Tuple[Model, Dict[str, Any], Dict[str, float]]:
            A tuple containing the best model, the best hyperparameters, and all scores.
        """
        raise NotImplementedError


class HyperparameterOptimizerValidation(HyperparameterOptimizer):
    """
    Provides simple grid hyperparameter search capabilities.
    This class performs a grid hyperparameter search over the specified
    hyperparameter space.
    """

    def hyperparameter_search(self,
                              params_dict: Dict,
                              train_dataset: Dataset,
                              valid_dataset: Dataset,
                              metric: Metric,
                              n_iter_search: int = 15,
                              n_jobs: int = 1,
                              verbose: int = 0,
                              use_max: bool = True,
                              logdir: str = None,
                              **kwargs):
        """
        Perform hyperparams search according to params_dict.
        Each key to hyperparams_dict is a model_param. The values should be a list of potential values for that
        hyperparameter.

        Parameters
        ----------
        params_dict: Dict
            Dictionary mapping hyperparameter names (strings) to lists of possible parameter values.
        train_dataset: Dataset
            The training dataset.
        valid_dataset: Dataset
            The validation dataset.
        metric: Metric
            The metric to optimize.
        n_iter_search: int
            Number of random combinations of parameters to test, if None performs complete grid search.
        n_jobs: int
            Number of jobs to run in parallel.
        verbose: int
            Controls the verbosity: the higher, the more messages.
        use_max: bool
            If True, return the model with the highest score.
        logdir: str
            The directory in which to store created models. If not set, will use a temporary directory.

        Returns
        -------
        Tuple[Model, Dict[str, Any], Dict[str, float]]:
            A tuple containing the best model, the best hyperparameters, and all scores.
        """
        if self.mode is None:
            # TODO: better way of doint this
            if len(set(train_dataset.y)) > 2:
                model = KerasRegressor(build_fn=self.model_builder, **kwargs)
                self.mode = 'regression'
            else:
                model = KerasClassifier(build_fn=self.model_builder, **kwargs)
                self.mode = 'classification'
        elif self.mode == 'classification':
            model = KerasClassifier(build_fn=self.model_builder, **kwargs)
        elif self.mode == 'regression':
            model = KerasRegressor(build_fn=self.model_builder, **kwargs)
        else:
            raise ValueError('Model operation mode can only be classification or regression!')

        print('MODE: ', self.mode)
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
        random_inds = random.sample(range(0, len_params), k=n_iter_search)
        best_hyperparams = None
        best_model, best_model_dir = None, None
        all_scores = {}
        j = 0
        print("Fitting %d random models from a space of %d possible models." % (
            len(random_inds), number_combinations))
        for ind, hyperparameter_tuple in enumerate(itertools.product(*hyperparam_vals)):
            if ind in random_inds:
                j += 1
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

                try:
                    model = SklearnModel(self.model_builder(**model_params), model_dir)

                except Exception as e:
                    model = KerasModel(self.model_builder(**model_params), model_dir)

                model.fit(train_dataset)

                try:
                    model.save()
                except Exception as e:
                    print(e)

                multitask_scores = model.evaluate(valid_dataset, [metric])[0]
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

        multitask_scores = best_model.evaluate(train_dataset, [metric])[0]
        train_score = multitask_scores[metric.name]
        print("Best hyperparameters: %s" % str(best_hyperparams))
        print("train_score: %f" % train_score)
        print("validation_score: %f" % best_validation_score)
        return best_model, best_hyperparams, all_scores


class HyperparameterOptimizerCV(HyperparameterOptimizer):
    """
    Provides simple grid hyperparameter search capabilities.
    This class performs a grid hyperparameter search over the specified
    hyperparameter space.
    """

    def hyperparameter_search(self,
                              model_type: str,
                              params_dict: Dict,
                              train_dataset: Dataset,
                              metric: str,
                              cv: int = 3,
                              n_iter_search: int = 15,
                              n_jobs: int = 1,
                              verbose: int = 0,
                              logdir: str = None,
                              seed: int = None,
                              **kwargs):

        """
        Perform hyperparams search according to params_dict.
        Each key to hyperparams_dict is a model_param. The values should be a list of potential values for that
        hyperparameter.

        Parameters
        ----------
        model_type: str
            Type of model to use. Must be one of 'sklearn', 'keras', and 'deepchem'.
        params_dict: Dict
            Dictionary mapping hyperparameter names (strings) to lists of possible parameter values.
        train_dataset: Dataset
            Dataset to train on.
        metric: Metric
            Metric to optimize over.
        cv: int
            Number of cross-validation folds to perform.
        n_iter_search: int
            Number of hyperparameter combinations to try.
        n_jobs: int
            Number of jobs to run in parallel.
        verbose: int
            Verbosity level.
        logdir: str
            The directory in which to store created models. If not set, will use a temporary directory.
        seed: int
            Random seed to use.
        kwargs: dict
            Additional keyword arguments to pass to the model constructor.

        Returns
        -------
        Tuple[Model, Dict[str, Any], Dict[str, float]]:
            A tuple containing the best model, the best hyperparameters, and all scores.
        """
        # TODO: better way of doing this
        if self.mode is None:
            if len(set(train_dataset.y)) > 2:
                self.mode = 'regression'
            else:
                self.mode = 'classification'

        if model_type != 'deepchem':
            if self.mode == 'classification':
                cv = StratifiedKFold(n_splits=cv, shuffle=True,
                                     random_state=seed)  # changed this so that we can set seed here and shuffle data
                # by default
            else:
                cv = KFold(n_splits=cv, shuffle=True,
                           random_state=seed)  # added this so that data is shuffled before splitting

        print('MODEL TYPE: ', model_type)
        # diferentiate sklearn model from keras model
        if model_type == 'keras':
            if self.mode == 'classification':
                model = KerasClassifier(build_fn=self.model_builder, **kwargs)
            elif self.mode == 'regression':
                model = KerasRegressor(build_fn=self.model_builder, **kwargs)
            else:
                raise ValueError('Model operation mode can only be classification or regression!')
        elif model_type == 'sklearn':
            model = self.model_builder()
            # model = SklearnModel(self.model_builder, self.mode)
        elif model_type == 'deepchem':
            model = self.model_builder  # because we don't want to call the function yet
        else:
            raise ValueError('Only keras, sklearn and deepchem models are accepted.')

        metrics = validate_metrics(metric)

        number_combinations = reduce(mul, [len(vals) for vals in params_dict.values()])
        if number_combinations > n_iter_search:
            print("Fitting %d random models from a space of %d possible models." % (n_iter_search, number_combinations))
            if model_type == 'deepchem':
                grid = DeepchemRandomSearchCV(model_build_fn=model, param_distributions=params_dict, scoring=metrics,
                                              cv=cv, mode=self.mode, n_iter=n_iter_search, refit=True,
                                              random_state=seed)
            else:
                grid = RandomizedSearchCV(estimator=model, param_distributions=params_dict, scoring=metrics,
                                          n_jobs=n_jobs, cv=cv, verbose=verbose, n_iter=n_iter_search, refit=False,
                                          random_state=seed)
        else:
            # if self.mode == 'classification': grid = GridSearchCV(estimator = model, param_grid = params_dict,
            # scoring = metrics, n_jobs=n_jobs, cv=StratifiedKFold(n_splits=cv), verbose=verbose, refit=False) else:
            # grid = RandomizedSearchCV(estimator = model, param_distributions = params_dict, scoring = metrics,
            # n_jobs=n_jobs, cv=cv, verbose=verbose, n_iter = n_iter_search, refit=False)
            if model_type == 'deepchem':
                grid = DeepchemGridSearchCV(model_build_fn=model, param_grid=params_dict, scoring=metrics,
                                            cv=cv, mode=self.mode, refit=True, random_state=seed)
            else:
                grid = GridSearchCV(estimator=model, param_grid=params_dict, scoring=metrics, n_jobs=n_jobs,
                                    cv=cv, verbose=verbose, refit=False)

        # print(train_dataset.X.shape, train_dataset.X.shape[0]/cv)
        if model_type == 'deepchem':
            grid.fit(train_dataset)
            grid_result = grid  # because fit here doesn't return the object
        else:
            grid_result = grid.fit(train_dataset.X, train_dataset.y)
        print("\n \n Best %s: %f using %s" % (metrics, grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("\n %s: %f (%f) with: %r \n" % (metrics, mean, stdev, param))

        if model_type == 'keras':
            best_model = KerasModel(self.model_builder, self.mode, **grid_result.best_params_)
            print('Fitting best model!')
            best_model.fit(train_dataset)
            return best_model, grid_result.best_params_, grid_result.cv_results_
        elif model_type == 'sklearn':
            best_model = SklearnModel(self.model_builder(**grid_result.best_params_), self.mode)
            print('Fitting best model!')
            best_model.fit(train_dataset)
            print(best_model)
            return best_model, grid_result.best_params_, grid_result.cv_results_
        else:  # DeepchemModel
            return grid_result.best_estimator_, grid_result.best_params_, grid_result.cv_results_
