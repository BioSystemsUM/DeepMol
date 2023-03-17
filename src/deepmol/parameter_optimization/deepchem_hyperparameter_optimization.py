"""Hyperparameter Optimization Classes for DeepchemModel models"""
from collections import defaultdict
from typing import Union, List

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler

from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.splitters import SingletaskStratifiedSplitter, RandomSplitter


# TODO: it would probably be better if we tried to create a scikit-learn wrapper for DeepchemModel models,
#  similar to KerasRegressor and KerasClassifier (not sure it would work though)
class DeepchemBaseSearchCV(object):
    """
    Base class for hyperparameter search with cross-validation for DeepChemModels.
    """

    def __init__(self,
                 model_build_fn: callable,
                 param_grid: Union[dict, ParameterGrid, ParameterSampler],
                 scoring: Union[Metric, List[Metric]],
                 maximize: bool,
                 refit: bool,
                 cv: int,
                 mode: str,
                 random_state: int = None,
                 return_train_score: bool = False):
        """
        Initialize the hyperparameter search.

        Parameters
        ----------
        model_build_fn: callable
            A function that builds a DeepchemModel model.
        param_grid: dict
            The hyperparameter grid to search.
        scoring: Union[Metric, List[Metric]]
            The metrics to use for scoring.
        maximize: bool
            If True, maximize the metric. If False, minimize the metric.
        refit: bool
            If True, refit the best model on the whole dataset.
        cv: int
            The number of folds for cross-validation.
        mode: str
            The mode of the model. Can be 'classification' or 'regression'.
        random_state: int
            The random state to use for the cross-validation.
        return_train_score: bool
            If True, return the training scores.
        """
        self.build_fn = model_build_fn
        self.param_grid = param_grid
        self.metric = scoring
        self.maximize = maximize
        self.mode = mode
        self.refit = refit
        self.cv = cv
        self.random_state = random_state
        self.return_train_score = return_train_score

        self.best_score_ = None
        self.best_params_ = {}
        self.best_estimator_ = None
        self.cv_results_ = None

    def fit(self, dataset: Dataset):
        """
        Run hyperparameter search with cross-validation.

        Parameters
        ----------
        dataset: Dataset
            The dataset to use for the hyperparameter search.
        """
        if self.mode != dataset.mode:
            raise ValueError(f'The mode of the model and the dataset must be the same. Got {self.mode} and '
                             f'{dataset.mode} respectively.')
        results_dict = defaultdict(list)

        # split dataset into folds
        if dataset.mode == 'classification':
            splitter = SingletaskStratifiedSplitter()
        else:
            splitter = RandomSplitter()

        datasets = splitter.k_fold_split(dataset, k=self.cv, seed=self.random_state)
        for param_combination in self.param_grid:
            results_dict['params'].append(param_combination)

            # Cross-validation:
            train_scores = []
            test_scores = []
            for train_dataset, test_dataset in datasets:
                model = self.build_fn(**param_combination)  # creates a new DeepchemModel
                model.fit(train_dataset)
                train_score, _ = model.evaluate(train_dataset, [self.metric])
                train_scores.append(train_score[self.metric.name])
                test_score, _ = model.evaluate(test_dataset, [self.metric])
                test_scores.append(test_score[self.metric.name])

            results_dict['mean_train_score'].append(np.mean(train_scores))
            mean_test_score = np.mean(test_scores)
            results_dict['mean_test_score'].append(mean_test_score)
            results_dict['std_train_score'].append(np.std(train_scores))
            results_dict['std_test_score'].append(np.std(test_scores))
            for i, (train_score, test_score) in enumerate(zip(train_scores, test_scores)):
                train_key = 'split%s_train_score' % str(i)
                test_key = 'split%s_test_score' % str(i)
                results_dict[train_key].append(train_score)
                results_dict[test_key].append(test_score)

            if self.maximize:
                if (self.best_score_ is None) or (mean_test_score > self.best_score_):
                    self.best_score_ = mean_test_score
                    self.best_params_ = param_combination
            else:
                if (self.best_score_ is None) or (mean_test_score < self.best_score_):
                    self.best_score_ = mean_test_score
                    self.best_params_ = param_combination

        self.cv_results_ = results_dict
        self.best_estimator_ = self.build_fn(**self.best_params_)

        if self.refit:
            self.best_estimator_.fit(dataset)


class DeepchemGridSearchCV(DeepchemBaseSearchCV):
    """
    Hyperparameter search with cross-validation for DeepChemModels using a grid search.
    """

    def __init__(self,
                 model_build_fn: callable,
                 param_grid: Union[dict, ParameterGrid],
                 scoring: Union[Metric, List[Metric]],
                 maximize: bool,
                 refit: bool,
                 cv: int,
                 mode: str,
                 random_state: int = None,
                 return_train_score: bool = False):
        """
        Initialize the hyperparameter search.

        Parameters
        ----------
        model_build_fn: callable
            A function that builds a DeepchemModel model.
        param_grid: Union[dict, ParameterGrid]
            The hyperparameter grid to search.
        scoring: Union[Metric, List[Metric]]
            The metric to use for scoring.
        maximize: bool
            If True, maximize the metric. If False, minimize the metric.
        refit: bool
            If True, refit the best model on the whole dataset.
        cv: int
            The number of folds for cross-validation.
        mode: str
            The mode of the model. Can be 'classification' or 'regression'.
        random_state: int
            The random state to use for the cross-validation.
        return_train_score: bool
            If True, return the training scores.
        """
        self.param_grid = ParameterGrid(param_grid)
        super().__init__(model_build_fn=model_build_fn, param_grid=self.param_grid, scoring=scoring, maximize=maximize,
                         refit=refit, cv=cv, mode=mode, random_state=random_state, return_train_score=return_train_score)


class DeepchemRandomSearchCV(DeepchemBaseSearchCV):
    """
    Hyperparameter search with cross-validation for DeepChemModels using a random search.
    """

    def __init__(self,
                 model_build_fn: callable,
                 param_distributions: Union[dict, ParameterSampler],
                 scoring: Union[Metric, List[Metric]],
                 maximize: bool,
                 refit: bool,
                 cv: int,
                 mode: str,
                 random_state: int = None,
                 return_train_score: bool = False,
                 n_iter: int = 20):
        """
        Initialize the hyperparameter search.

        Parameters
        ----------
        model_build_fn: callable
            A function that builds a DeepchemModel model.
        param_distributions: Union[dict, ParameterSampler]
            The hyperparameter sampler to search.
        scoring: Union[Metric, List[Metric]]
            The metrics to use for scoring.
        maximize: bool
            If True, maximize the metric. If False, minimize the metric.
        refit: bool
            If True, refit the best model on the whole dataset.
        cv: int
            The number of folds for cross-validation.
        mode: str
            The mode of the model. Can be 'classification' or 'regression'.
        random_state: int
            The random state to use for the cross-validation.
        return_train_score: bool
            If True, return the training scores.
        n_iter: int
            The number of iterations to perform.
        """
        self.param_grid = ParameterSampler(param_distributions, n_iter, random_state=random_state)
        super().__init__(model_build_fn=model_build_fn, param_grid=self.param_grid, scoring=scoring, maximize=maximize,
                         refit=refit, cv=cv, mode=mode, random_state=random_state, return_train_score=return_train_score)
