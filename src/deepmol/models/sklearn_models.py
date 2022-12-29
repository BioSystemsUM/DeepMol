from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator

from deepmol.models.models import Model
from deepmol.datasets import Dataset
from deepmol.splitters.splitters import RandomSplitter, SingletaskStratifiedSplitter
from deepmol.metrics.metrics import Metric

from deepmol.utils.utils import load_from_disk, save_to_disk

from sklearn.base import clone


class SklearnModel(Model):
    """
    Wrapper class that wraps scikit-learn models.
    The `SklearnModel` class provides a wrapper around scikit-learn models that allows scikit-learn models to be
    trained on `Dataset` objects and evaluated with the metrics in Metrics.
    """

    def __init__(self, model: BaseEstimator, mode: str = None, model_dir: str = None, **kwargs):
        """
        Initializes a `SklearnModel` object.

        Parameters
        ----------
        model: BaseEstimator
          The model instance which inherits a scikit-learn `BaseEstimator` Class.
        mode: str
            'classification' or 'regression'
        model_dir: str
          If specified the model will be stored in this directory. Else, a temporary directory will be used.
        kwargs: dict
            Additional keyword arguments.
        """
        super().__init__(model, model_dir, **kwargs)
        self.mode = mode
        self.model_type = 'sklearn'

    def fit_on_batch(self, X: Sequence, y: Sequence):
        """
        Fits model on batch of data.
        """
        raise NotImplementedError

    def get_task_type(self) -> str:
        """
        Returns the task type of the model.
        """
        raise NotImplementedError

    def get_num_tasks(self) -> int:
        """
        Returns the number of tasks.
        """
        raise NotImplementedError

    def fit(self, dataset: Dataset) -> None:
        """
        Fits scikit-learn model to data.

        Parameters
        ----------
        dataset: Dataset
            The `Dataset` to train this model on.

        Returns
        -------
        BaseEstimator
            The trained scikit-learn model.
        """
        features = dataset.X
        y = np.squeeze(dataset.y)
        return self.model.fit(features, y)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Makes predictions on dataset.

        Parameters
        ----------
        dataset: Dataset
          Dataset to make prediction on.

        Returns
        -------
        np.ndarray
          The value is a return value of `predict_proba` or `predict` method of the scikit-learn model. If the
          scikit-learn model has both methods, the value is always a return value of `predict_proba`.
        """
        try:
            return self.model.predict_proba(dataset.X)
        except AttributeError:
            return self.model.predict(dataset.X)

    def predict_on_batch(self, dataset: Dataset) -> np.ndarray:
        """
        Makes predictions on batch of data.

        Parameters
        ----------
        dataset: Dataset
          Dataset to make prediction on.

        Returns
        -------
        np.ndarray
            numpy array of predictions.
        """
        return super(SklearnModel, self).predict(dataset)

    def save(self):
        """
        Saves scikit-learn model to disk using joblib.
        """
        save_to_disk(self.model, self.get_model_filename(self.model_dir))

    def reload(self):
        """
        Loads scikit-learn model from joblib file on disk.
        """
        self.model = load_from_disk(self.get_model_filename(self.model_dir))

    def cross_validate(self,
                       dataset: Dataset,
                       metric: Metric,
                       folds: int = 3):
        """
        Performs cross-validation on a dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to perform cross-validation on.
        metric: Metric
            Metric to evaluate model performance.
        folds: int
            Number of folds to use for cross-validation.

        Returns
        -------
        Tuple[SKlearnModel, float, float, List[float], List[float], float, float]
            The first element is the best model, the second is the train score of the best model, the third is the train
            score of the best model, the fourth is the test scores of all models, the fifth is the average train scores
            of all folds and the sixth is the average test score of all folds.
        """
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
            train_score = dummy_model.evaluate(train_ds, metric)[0]
            train_scores.append(train_score[metric.name])
            avg_train_score += train_score[metric.name]

            print('Test Score: ')
            test_score = dummy_model.evaluate(test_ds, metric)[0]
            test_scores.append(test_score[metric.name])
            avg_test_score += test_score[metric.name]

            if test_score[metric.name] > test_score_best_model:
                test_score_best_model = test_score[metric.name]
                train_score_best_model = train_score[metric.name]
                best_model = dummy_model

        return best_model, train_score_best_model, test_score_best_model, train_scores, test_scores, avg_train_score / folds, avg_test_score / folds
