from typing import Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator

from models.models import Model
from datasets.datasets import Dataset
from splitters.splitters import RandomSplitter, SingletaskStratifiedSplitter
from metrics.metrics import Metric

from utils.utils import load_from_disk, save_to_disk

from sklearn.base import clone

'''
#Some ScikitLearn non weighted models
NON_WEIGHTED_MODELS = [
    LogisticRegression, PLSRegression, GaussianProcessRegressor, ElasticNetCV,
    LassoCV, BayesianRidge
]
'''


class SklearnModel(Model):
    """Wrapper class that wraps scikit-learn models.
    The `SklearnModel` class provides a wrapper around scikit-learn
    models that allows scikit-learn models to be trained on `Dataset` objects
    and evaluated with the metrics in Metrics.
    """

    def fit_on_batch(self, X: Sequence, y: Sequence):
        pass

    def get_task_type(self) -> str:
        pass

    def get_num_tasks(self) -> int:
        pass

    def __init__(self, model: BaseEstimator, mode: Optional[str] = None, model_dir: Optional[str] = None, **kwargs):
        """
        Parameters
        ----------
        model: BaseEstimator
          The model instance which inherits a scikit-learn `BaseEstimator` Class.
        mode: str, optional (default None)
            'classification' or 'regression'
        model_dir: str, optional (default None)
          If specified the model will be stored in this directory. Else, a
          temporary directory will be used.
        kwargs: dict
        """

        super().__init__(model, model_dir, **kwargs)
        self.model = model
        self.mode = mode
        self.model_dir = model_dir
        self.model_type = 'sklearn'

    def fit(self, dataset: Dataset) -> None:
        """Fits scikit-learn model to data.
        Parameters
        ----------
        dataset: Dataset
            The `Dataset` to train this model on.
        """
        features = dataset.X
        y = np.squeeze(dataset.y)
        '''
        # Some scikit-learn models don't use weights.
        if self.use_weights:
            self.model.fit(features, y)
            return
        self.model.fit(features, y)
        '''
        return self.model.fit(features, y)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """Makes predictions on dataset.
        Parameters
        ----------
        dataset: Dataset
          Dataset to make prediction on.
        Returns
        -------
        np.ndarray
          The value is a return value of `predict_proba` or `predict` method
          of the scikit-learn model. If the scikit-learn model has both methods,
          the value is always a return value of `predict_proba`.
        """
        try:
            return self.model.predict_proba(dataset.X)
        except AttributeError:
            return self.model.predict(dataset.X)

    def predict_on_batch(self, dataset: Dataset) -> np.ndarray:
        """Makes predictions on batch of data.
        Parameters
        ----------
        dataset: Dataset
          Dataset to make prediction on.
        """
        return super(SklearnModel, self).predict(dataset)

    def save(self):
        """Saves scikit-learn model to disk using joblib."""
        save_to_disk(self.model, self.get_model_filename(self.model_dir))

    def reload(self):
        """Loads scikit-learn model from joblib file on disk."""
        self.model = load_from_disk(self.get_model_filename(self.model_dir))

    def cross_validate(self,
                       dataset: Dataset,
                       metric: Metric,
                       folds: int = 3):

        # TODO: add option to choose between splitters
        if self.mode == 'classification':
            splitter = SingletaskStratifiedSplitter()
            datasets = splitter.k_fold_split(dataset, folds)
        elif self.mode == 'regression':
            splitter = RandomSplitter()
            datasets = splitter.k_fold_split(dataset, folds)
        else:
            try:
                splitter = SingletaskStratifiedSplitter()
                datasets = splitter.k_fold_split(dataset, folds)
            except Exception as e:
                splitter = RandomSplitter()
                datasets = splitter.k_fold_split(dataset, folds)

        train_scores = []
        train_score_best_model = 0
        avg_train_score = 0

        test_scores = []
        test_score_best_model = 0
        avg_test_score = 0
        best_model = None
        split = 1
        for train_ds, test_ds in datasets:
            print('\nSplit', str(split), ':')
            split += 1
            dummy_model = clone(SklearnModel(model=self.model))

            dummy_model.fit(train_ds)

            print('Train Score: ')
            train_score = dummy_model.evaluate(train_ds, metric)
            train_scores.append(train_score[metric.name])
            avg_train_score += train_score[metric.name]

            print('Test Score: ')
            test_score = dummy_model.evaluate(test_ds, metric)
            test_scores.append(test_score[metric.name])
            avg_test_score += test_score[metric.name]

            if test_score[metric.name] > test_score_best_model:
                test_score_best_model = test_score[metric.name]
                train_score_best_model = train_score[metric.name]
                best_model = dummy_model

        return best_model, train_score_best_model, test_score_best_model, train_scores, test_scores, avg_train_score / folds, avg_test_score / folds
