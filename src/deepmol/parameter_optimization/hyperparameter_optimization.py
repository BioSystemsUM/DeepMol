import itertools
import os
import random
import shutil
import tempfile
from functools import reduce
from operator import mul
from typing import Dict, Any, Tuple

import numpy as np
try:
    from scikeras.wrappers import KerasRegressor, KerasClassifier
except ImportError:
    pass
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV, GridSearchCV

from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.models import SklearnModel, KerasModel
from deepmol.models.models import Model
from deepmol.parameter_optimization._utils import _convert_hyperparam_dict_to_filename
from deepmol.parameter_optimization.base_hyperparameter_optimization import HyperparameterOptimizer
from deepmol.parameter_optimization.deepchem_hyperparameter_optimization import DeepchemRandomSearchCV, \
    DeepchemGridSearchCV


class HyperparameterOptimizerValidation(HyperparameterOptimizer):
    """
    Provides simple grid hyperparameter search capabilities.
    This class performs a grid hyperparameter search over the specified
    hyperparameter space.
    """

    def fit(self, train_dataset: Dataset, valid_dataset: Dataset = None) \
            -> Tuple[Model, Dict[str, Any], Dict[str, float]]:
        """
        Perform hyperparams search according to params_dict.
        Each key to hyperparams_dict is a model_param. The values should be a list of potential values for that
        hyperparameter.

        Parameters
        ----------

        train_dataset: Dataset
            The training dataset.
        valid_dataset: Dataset
            The validation dataset.


        Returns
        -------
        Tuple[Model, Dict[str, Any], Dict[str, float]]:
            A tuple containing the best model, the best hyperparameters, and all scores.
        """
        assert valid_dataset is not None, "Validation dataset must be provided for this mode of " \
                                          "hyperparameter optimization."

        if self.mode is None:
            self.mode = train_dataset.mode
        else:
            if self.mode != train_dataset.mode:
                raise ValueError(f'Train dataset mode does not match model operation mode! Got {train_dataset.mode} '
                                 f'but expected {self.mode}')

        hyperparams = self.params_dict.keys()
        hyperparameter_values = self.params_dict.values()

        number_combinations = reduce(mul, [len(vals) for vals in hyperparameter_values])

        if self.maximize_metric:
            best_validation_score = -np.inf
        else:
            best_validation_score = np.inf

        # To make sure that the number of iterations is lower or equal to the number of max hypaparameter combinations
        len_params = sum(1 for x in itertools.product(*self.params_dict.values()))
        if self.n_iter_search is None or len_params < self.n_iter_search:
            n_iter_search = len_params
        else:
            n_iter_search = self.n_iter_search
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

                if self.logdir is not None:
                    model_dir = os.path.join(self.logdir, str(ind))
                    self.logger.info("model_dir is %s" % model_dir)
                    try:
                        os.makedirs(model_dir)
                    except OSError:
                        if not os.path.isdir(model_dir):
                            self.logger.error("Error creating model_dir, using tempfile directory")
                            model_dir = tempfile.mkdtemp()
                else:
                    model_dir = tempfile.mkdtemp()

                if self.model_type == 'sklearn':
                    model = SklearnModel(model=self.model_builder(**model_params),
                                         mode=self.mode,
                                         model_dir=model_dir)
                    model.fit(train_dataset)

                elif self.model_type == 'keras':
                    model = KerasModel(model_builder=self.model_builder,
                                       mode=self.mode,
                                       model_dir=model_dir,
                                       **model_params)
                    model.fit(train_dataset)
                elif self.model_type == 'deepchem':
                    model = self.model_builder(**model_params)
                    model.fit(train_dataset)

                try:
                    model.save()
                except Exception as e:
                    self.logger.error(str(e))

                multitask_scores = model.evaluate(valid_dataset, [self.metric])[0]
                valid_score = multitask_scores[self.metric.name]
                hp_str = _convert_hyperparam_dict_to_filename(hyper_params)
                all_scores[hp_str] = valid_score

                if (self.maximize_metric and valid_score >= best_validation_score) or (
                        not self.maximize_metric and valid_score <= best_validation_score):
                    best_validation_score = valid_score
                    best_hyperparams = hyper_params
                    if best_model_dir is not None:
                        shutil.rmtree(best_model_dir)
                    best_model_dir = model_dir
                    best_model = model
                else:
                    shutil.rmtree(model_dir)

                self.logger.info("Model %d/%d, Metric %s, Validation set %s: %f" % (
                    j, len(random_indexes), self.metric.name, j, valid_score))
                self.logger.info("\tbest_validation_score so far: %f" % best_validation_score)

        if best_model is None:
            self.logger.warning("No models trained correctly.")
            # arbitrarily return last model
            best_model, best_hyperparams = model, hyper_params
            return best_model, best_hyperparams, all_scores

        multitask_scores = best_model.evaluate(train_dataset, [self.metric])[0]
        train_score = multitask_scores[self.metric.name]
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

    def __init__(self, model_builder: callable, model_type: str, params_dict: Dict, metric: Metric,
                 maximize_metric: bool, n_iter_search: int = 15, n_jobs: int = 1, verbose: int = 0, logdir: str = None,
                 mode: str = None, cv=5, seed=123, refit=True, **kwargs):

        if model_type == 'keras':
            # get random values from the params_dict
            # just to make sure that we instantiate the parameters in the keras model
            keras_default_parameters = {k: [random.choice(v)] for k, v in params_dict.items()}
            kwargs.update(keras_default_parameters)

        super().__init__(model_builder, model_type, params_dict, metric, maximize_metric, n_iter_search, n_jobs,
                         verbose, logdir, mode, **kwargs)
        
        self.cv = cv
        self.seed = seed
        self.refit = refit

    def fit(self, train_dataset: Dataset, validation_dataset: Dataset = None) -> Tuple[Model, Dict, Dict]:

        """
        Perform hyperparams search according to params_dict.
        Each key to hyperparams_dict is a model_param. The values should be a list of potential values for that
        hyperparameter.

        Parameters
        ----------
        train_dataset: Dataset
            Dataset to train on.
        validation_dataset: Dataset, optional
            Dataset to validate on.

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

        if self.model_type != 'deepchem':
            if self.mode == 'classification':
                folds = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                     random_state=self.seed)  # changed this so that we can set seed here and shuffle data
                # by default
            else:
                folds = KFold(n_splits=self.cv, shuffle=True,
                           random_state=self.seed)  # added this so that data is shuffled before splitting

        self.logger.info(f'MODEL TYPE: {self.model_type}')
        # diferentiate sklearn model from keras model
        if self.model_type == 'keras':
            if self.mode == 'classification':
                model = KerasClassifier(build_fn=self.model_builder, **self.kwargs)
            elif self.mode == 'regression':
                model = KerasRegressor(build_fn=self.model_builder, **self.kwargs)
            else:
                raise ValueError('Model operation mode can only be classification or regression!')
        elif self.model_type == 'sklearn':
            model = self.model_builder()
        elif self.model_type == 'deepchem':
            model = self.model_builder  # because we don't want to call the function yet
        else:
            raise ValueError('Only keras, sklearn and deepchem models are accepted.')

        number_combinations = reduce(mul, [len(vals) for vals in self.params_dict.values()])
        if number_combinations > self.n_iter_search:
            self.logger.info("Fitting %d random models from a space of %d possible models." % (self.n_iter_search,
                                                                                               number_combinations))
            if self.model_type == 'deepchem':
                grid = DeepchemRandomSearchCV(model_builder=model, params_dict=self.params_dict,
                                              metric=self.metric,
                                              maximize_metric=self.maximize_metric,
                                              cv=self.cv, mode=self.mode,
                                              n_iter=self.n_iter_search,
                                              refit=self.refit, random_state=self.seed)
            else:
                grid = RandomizedSearchCV(estimator=model, param_distributions=self.params_dict,
                                          scoring=make_scorer(self.metric.metric),
                                          n_jobs=self.n_jobs, cv=folds,
                                          verbose=self.verbose, n_iter=self.n_iter_search,
                                          refit=self.refit,
                                          random_state=self.seed, error_score='raise')
        else:
            if self.model_type == 'deepchem':
                grid = DeepchemGridSearchCV(model_builder=model, params_dict=self.params_dict, metric=self.metric,
                                            maximize_metric=self.maximize_metric, cv=self.cv,
                                            mode=self.mode, refit=self.refit,
                                            random_state=self.seed)
            else:
                grid = GridSearchCV(estimator=model, param_grid=self.params_dict,
                                    scoring=make_scorer(self.metric.metric),
                                    n_jobs=self.n_jobs,
                                    cv=folds, verbose=self.verbose, refit=self.refit, error_score='raise')

        if self.model_type == 'deepchem':
            grid.fit(train_dataset)
            grid_result = grid  # because fit here doesn't return the object
        else:
            grid_result = grid.fit(train_dataset.X, train_dataset.y)

        self.logger.info("\n \n Best %s: %f using %s" % (self.metric.metric, grid_result.best_score_,
                                                         grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            self.logger.info("\n %s: %f (%f) with: %r \n" % (self.metric.metric, mean, stdev, param))

        if self.model_type == 'keras':
            best_model = KerasModel(self.model_builder, self.mode, **grid_result.best_params_)
            self.logger.info('Fitting best model!')
            best_model.fit(train_dataset)
            return best_model, grid_result.best_params_, grid_result.cv_results_
        elif self.model_type == 'sklearn':
            best_model = SklearnModel(self.model_builder(**grid_result.best_params_), self.mode)
            self.logger.info('Fitting best model!')
            best_model.fit(train_dataset)
            self.logger.info(str(best_model))
            return best_model, grid_result.best_params_, grid_result.cv_results_
        else:  # DeepchemModel
            return grid_result.best_estimator_, grid_result.best_params_, grid_result.cv_results_
