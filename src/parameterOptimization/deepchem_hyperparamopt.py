"""Hyperparameter Optimization Classes for DeepchemModel models"""

from collections import defaultdict
import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics._scorer import SCORERS
from metrics.Metrics import Metric
from splitters.splitters import RandomSplitter, SingletaskStratifiedSplitter


# TODO: it would probably be better if we tried to create a scikit-learn wrapper for DeepchemModel models,
#  similar to KerasRegressor and KerasClassifier (not sure it would work though)
class DeepchemBaseSearchCV(object):

    def __init__(self, model_build_fn, param_grid, scoring, refit, cv, mode, return_train_score=False):
        self.build_fn = model_build_fn
        self.param_grid = param_grid
        # TODO: it would be easier if we could just pass a Metric object instead
        if scoring in SCORERS.keys():
            scorer = SCORERS[scoring]
        else:
            scorer = scoring
        score_func = scorer._score_func
        #kwargs = scorer._kwargs
        self.metric = Metric(score_func, mode=mode)
        if 'error' in score_func.__name__:
            self.use_max = False
        else:
            self.use_max = True
        self.mode = mode
        self.refit = refit
        self.cv = cv
        self.return_train_score = return_train_score

        self.best_score_ = None
        self.best_params_ = {}
        self.best_estimator_ = None
        self.cv_results_ = None

    def fit(self, dataset):
        results_dict = defaultdict(list)

        # split dataset into folds
        if self.mode == 'classification':
            splitter = SingletaskStratifiedSplitter()
        else:
            splitter = RandomSplitter()

        datasets = splitter.k_fold_split(dataset, self.cv)
        for param_combination in self.param_grid:
            results_dict['params'].append(param_combination)

            # Cross-validation:
            train_scores = []
            test_scores = []
            for train_dataset, test_dataset in datasets:
                model = self.build_fn(**param_combination)  # creates a new DeepchemModel
                model.fit(train_dataset)
                train_score = model.evaluate(train_dataset, [self.metric])
                train_scores.append(train_score[self.metric.name])
                test_score = model.evaluate(test_dataset, [self.metric])
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

            if self.use_max:
                if (self.best_score_ is None) or (mean_test_score > self.best_score_):
                    self.best_score_ = mean_test_score
                    self.best_params_ = param_combination
            else:
                if (self.best_score_ is None) or (mean_test_score < self.best_score_):
                    self.best_score_ = mean_test_score
                    self.best_params_ = param_combination

        self.best_estimator_ = self.build_fn(**self.best_params_) # self.build_fn returns a DeepchemModel instance
        self.cv_results_ = results_dict


class DeepchemGridSearchCV(DeepchemBaseSearchCV):

    def __init__(self, model_build_fn, param_grid, scoring, refit, cv, mode, return_train_score=False):
        self.param_grid = ParameterGrid(param_grid)
        super().__init__(model_build_fn=model_build_fn, param_grid=self.param_grid, scoring=scoring, refit=refit, cv=cv,
                         mode=mode, return_train_score=return_train_score)


class DeepchemRandomSearchCV(DeepchemBaseSearchCV):

    def __init__(self, model_build_fn, param_distributions, scoring, refit, cv, mode, return_train_score=False,
                 n_iter=20, random_state=None):
        self.param_grid = list(ParameterSampler(param_distributions, n_iter, random_state))
        super().__init__(model_build_fn=model_build_fn, param_grid=self.param_grid, scoring=scoring, refit=refit, cv=cv,
                         mode=mode, return_train_score=return_train_score)