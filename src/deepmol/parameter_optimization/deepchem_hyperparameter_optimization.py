from collections import defaultdict
from typing import Union, List

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler

from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.parameter_optimization.base_hyperparameter_optimization import HyperparameterOptimizer
from deepmol.splitters import SingletaskStratifiedSplitter, RandomSplitter


# TODO: it would probably be better if we tried to create a scikit-learn wrapper for DeepchemModel models,
#  similar to KerasRegressor and KerasClassifier (not sure it would work though)
class DeepchemBaseSearchCV(HyperparameterOptimizer):
    """
    Base class for hyperparameter search with cross-validation for DeepChemModels.
    """

    def __init__(self, model_builder: callable,
                 params_dict: Union[dict, ParameterGrid, ParameterSampler],
                 metric: Union[Metric, List[Metric]],
                 maximize_metric: bool,
                 refit: bool,
                 cv: int,
                 mode: str,
                 random_state: int = None,
                 return_train_score: bool = False, **kwargs):
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
        super().__init__(model_builder, "deepchem", params_dict, metric, maximize_metric, **kwargs)
        self.metric = metric
        self.mode = mode
        self.refit = refit
        self.cv = cv
        self.random_state = random_state
        self.return_train_score = return_train_score

        self.best_score_ = None
        self.best_params_ = {}
        self.best_estimator_ = None
        self.cv_results_ = None

    def fit(self, train_dataset: Dataset, validation_dataset: Dataset = None):
        """
        Run hyperparameter search with cross-validation.

        Parameters
        ----------
        train_dataset: Dataset
            The dataset to use for the hyperparameter search.
        validation_dataset: Dataset
            The dataset to use for validation. If None, use the training dataset.
        """
        if self.mode != train_dataset.mode:
            raise ValueError(f'The mode of the model and the dataset must be the same. Got {self.mode} and '
                             f'{train_dataset.mode} respectively.')
        results_dict = defaultdict(list)

        # split dataset into folds
        if train_dataset.mode == 'classification':
            splitter = SingletaskStratifiedSplitter()
        else:
            splitter = RandomSplitter()

        datasets = splitter.k_fold_split(train_dataset, k=self.cv, seed=self.random_state)
        for param_combination in self.params_dict:
            results_dict['params'].append(param_combination)

            # Cross-validation:
            train_scores = []
            test_scores = []
            for train_dataset, test_dataset in datasets:
                model = self.model_builder(**param_combination)  # creates a new DeepchemModel
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

            if self.maximize_metric:
                if (self.best_score_ is None) or (mean_test_score > self.best_score_):
                    self.best_score_ = mean_test_score
                    self.best_params_ = param_combination
            else:
                if (self.best_score_ is None) or (mean_test_score < self.best_score_):
                    self.best_score_ = mean_test_score
                    self.best_params_ = param_combination

        self.cv_results_ = results_dict
        self.best_estimator_ = self.model_builder(**self.best_params_)

        if self.refit:
            self.best_estimator_.fit(train_dataset)

        return self.best_estimator_, self.best_params_, self.cv_results_


class DeepchemGridSearchCV(DeepchemBaseSearchCV):
    """
    Hyperparameter search with cross-validation for DeepChemModels using a grid search.
    """

    def __init__(self,
                 model_builder: callable,
                 params_dict: Union[dict, ParameterGrid],
                 metric: Union[Metric, List[Metric]],
                 maximize_metric: bool,
                 refit: bool,
                 cv: int,
                 mode: str,
                 random_state: int = None,
                 return_train_score: bool = False):
        """
        Initialize the hyperparameter search.

        Parameters
        ----------
        model_builder: callable
            A function that builds a DeepchemModel model.
        params_dict: Union[dict, ParameterGrid]
            The hyperparameter grid to search.
        metric: Union[Metric, List[Metric]]
            The metric to use for scoring.
        maximize_metric: bool
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
        self.param_grid = ParameterGrid(params_dict)
        super().__init__(model_builder=model_builder, params_dict=self.param_grid, metric=metric,
                         maximize_metric=maximize_metric,
                         refit=refit, cv=cv,
                         mode=mode, random_state=random_state,
                         return_train_score=return_train_score)


class DeepchemRandomSearchCV(DeepchemBaseSearchCV):
    """
    Hyperparameter search with cross-validation for DeepChemModels using a random search.
    """

    def __init__(self,
                 model_builder: callable,
                 params_dict: Union[dict, ParameterSampler],
                 metric: Union[Metric, List[Metric]],
                 maximize_metric: bool,
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
        model_builder: callable
            A function that builds a DeepchemModel model.
        param_grid: Union[dict, ParameterSampler]
            The hyperparameter sampler to search.
        metric: Union[Metric, List[Metric]]
            The metrics to use for scoring.
        maximize_metric: bool
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
        self.param_grid = ParameterSampler(params_dict, n_iter, random_state=random_state)
        super().__init__(model_builder=model_builder, params_dict=self.param_grid, metric=metric,
                         maximize_metric=maximize_metric,
                         refit=refit, cv=cv, mode=mode,
                         random_state=random_state, return_train_score=return_train_score)
