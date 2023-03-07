import itertools
import os
import random
import shutil
import tempfile
from abc import abstractmethod
from functools import reduce
from operator import mul
from typing import Dict, Any, Tuple, List, Union

import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV, GridSearchCV

from deepmol.datasets import Dataset
from deepmol.loggers.logger import Logger
from deepmol.metrics import Metric
from deepmol.models import SklearnModel, KerasModel
from deepmol.models.models import Model
from deepmol.parameter_optimization._utils import _convert_hyperparam_dict_to_filename, validate_metrics
from deepmol.parameter_optimization.deepchem_hyperparameter_optimization import DeepchemRandomSearchCV, \
    DeepchemGridSearchCV


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

        self.logger = Logger()

    @abstractmethod
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
            self.mode = train_dataset.mode
        else:
            if self.mode != train_dataset.mode:
                raise ValueError(f'Train dataset mode does not match model operation mode! Got {train_dataset.mode} '
                                 f'but expected {self.mode}')

        self.logger.info(f'MODE: {self.mode}')
        hyperparams = params_dict.keys()
        hyperparameter_values = params_dict.values()

        number_combinations = reduce(mul, [len(vals) for vals in hyperparameter_values])

        if use_max:
            best_validation_score = -np.inf
        else:
            best_validation_score = np.inf

        # To make sure that the number of iterations is lower or equal to the number of max hypaparameter combinations
        len_params = sum(1 for x in itertools.product(*params_dict.values()))
        if n_iter_search is None or len_params < n_iter_search:
            n_iter_search = len_params
        random_indexes = random.sample(range(0, len_params), k=n_iter_search)
        best_hyperparams = None
        best_model, best_model_dir = None, None
        all_scores = {}
        j = 0
        self.logger.info("Fitting %d random models from a space of %d possible models." % (
            len(random_indexes), number_combinations))
        for ind, hyperparameter_tuple in enumerate(itertools.product(*hyperparameter_values)):
            if ind in random_indexes:
                j += 1
                model_params = {}
                self.logger.info("Fitting model %d/%d" % (j, len(random_indexes)))
                hyper_params = dict(zip(hyperparams, hyperparameter_tuple))
                for hyperparameter, hyperparameter_value in zip(hyperparams, hyperparameter_tuple):
                    model_params[hyperparameter] = hyperparameter_value
                self.logger.info("hyperparameters: %s" % str(model_params))

                if logdir is not None:
                    model_dir = os.path.join(logdir, str(ind))
                    self.logger.info("model_dir is %s" % model_dir)
                    try:
                        os.makedirs(model_dir)
                    except OSError:
                        if not os.path.isdir(model_dir):
                            self.logger.error("Error creating model_dir, using tempfile directory")
                            model_dir = tempfile.mkdtemp()
                else:
                    model_dir = tempfile.mkdtemp()

                try:
                    model = SklearnModel(model=self.model_builder(**model_params),
                                         mode=self.mode,
                                         model_dir=model_dir)

                except Exception as e:
                    model = KerasModel(model_builder=self.model_builder(**model_params),
                                       mode=self.mode,
                                       model_dir=model_dir)

                model.fit(train_dataset)

                try:
                    model.save()
                except Exception as e:
                    self.logger.error(str(e))

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

                self.logger.info("Model %d/%d, Metric %s, Validation set %s: %f" % (
                    j, len(random_indexes), metric.name, j, valid_score))
                self.logger.info("\tbest_validation_score so far: %f" % best_validation_score)

        if best_model is None:
            self.logger.warning("No models trained correctly.")
            # arbitrarily return last model
            best_model, best_hyperparams = model, hyperparameter_tuple
            return best_model, best_hyperparams, all_scores

        multitask_scores = best_model.evaluate(train_dataset, [metric])[0]
        train_score = multitask_scores[metric.name]
        self.logger.info("Best hyperparameters: %s" % str(best_hyperparams))
        self.logger.info("train_score: %f" % train_score)
        self.logger.info("validation_score: %f" % best_validation_score)
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
                              metric: Union[str, Metric, callable],
                              cv: int = 3,
                              n_iter_search: int = 15,
                              n_jobs: int = 1,
                              verbose: int = 0,
                              logdir: str = None,
                              seed: int = None,
                              refit: bool = True,
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
        metric: Union[str, List[str], Metric, List[Metric], Dict]
            Metric or metrics to optimize over.
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
        refit: bool
            Whether to refit the best model on the entire training set.
        kwargs: dict
            Additional keyword arguments to pass to the model constructor.

        Returns
        -------
        Tuple[Model, Dict[str, Any], Dict[str, float]]:
            A tuple containing the best model, the best hyperparameters, and all scores.
        """
        if self.mode is None:
            self.mode = train_dataset.mode
        else:
            if self.mode != train_dataset.mode:
                raise ValueError(f'Train dataset mode does not match model operation mode! Got {train_dataset.mode} '
                                 f'but expected {self.mode}')

        if model_type != 'deepchem':
            if self.mode == 'classification':
                cv = StratifiedKFold(n_splits=cv, shuffle=True,
                                     random_state=seed)  # changed this so that we can set seed here and shuffle data
                # by default
            else:
                cv = KFold(n_splits=cv, shuffle=True,
                           random_state=seed)  # added this so that data is shuffled before splitting

        self.logger.info(f'MODEL TYPE: {model_type}')
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
        elif model_type == 'deepchem':
            model = self.model_builder  # because we don't want to call the function yet
        else:
            raise ValueError('Only keras, sklearn and deepchem models are accepted.')

        metric = validate_metrics(metric)

        number_combinations = reduce(mul, [len(vals) for vals in params_dict.values()])
        if number_combinations > n_iter_search:
            self.logger.info("Fitting %d random models from a space of %d possible models." % (n_iter_search,
                                                                                               number_combinations))
            if model_type == 'deepchem':
                grid = DeepchemRandomSearchCV(model_build_fn=model, param_distributions=params_dict, scoring=metric,
                                              cv=cv, mode=self.mode, n_iter=n_iter_search, refit=refit,
                                              random_state=seed)
            else:
                grid = RandomizedSearchCV(estimator=model, param_distributions=params_dict, scoring=metric,
                                          n_jobs=n_jobs, cv=cv, verbose=verbose, n_iter=n_iter_search, refit=refit,
                                          random_state=seed)
        else:
            if model_type == 'deepchem':
                grid = DeepchemGridSearchCV(model_build_fn=model, param_grid=params_dict, scoring=metric,
                                            cv=cv, mode=self.mode, refit=refit, random_state=seed)
            else:
                grid = GridSearchCV(estimator=model, param_grid=params_dict, scoring=metric, n_jobs=n_jobs,
                                    cv=cv, verbose=verbose, refit=refit)

        if model_type == 'deepchem':
            grid.fit(train_dataset)
            grid_result = grid  # because fit here doesn't return the object
        else:
            grid_result = grid.fit(train_dataset.X, train_dataset.y)

        self.logger.info("\n \n Best %s: %f using %s" % (metric, grid_result.best_score_,
                                                         grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            self.logger.info("\n %s: %f (%f) with: %r \n" % (metric, mean, stdev, param))

        if model_type == 'keras':
            best_model = KerasModel(self.model_builder, self.mode, **grid_result.best_params_)
            self.logger.info('Fitting best model!')
            best_model.fit(train_dataset)
            return best_model, grid_result.best_params_, grid_result.cv_results_
        elif model_type == 'sklearn':
            best_model = SklearnModel(self.model_builder(**grid_result.best_params_), self.mode)
            self.logger.info('Fitting best model!')
            best_model.fit(train_dataset)
            self.logger.info(str(best_model))
            return best_model, grid_result.best_params_, grid_result.cv_results_
        else:  # DeepchemModel
            return grid_result.best_estimator_, grid_result.best_params_, grid_result.cv_results_
