import os

import numpy as np
from sklearn.base import BaseEstimator

from deepmol.models._utils import save_to_disk, _get_splitter, load_from_disk
from deepmol.models.models import Model
from deepmol.datasets import Dataset
from deepmol.splitters.splitters import Splitter
from deepmol.metrics.metrics import Metric

from sklearn.base import clone

from deepmol.utils.utils import normalize_labels_shape


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
        self.parameters_to_save = {'mode': self.mode}

    @property
    def model_type(self):
        """
        Returns the type of the model.
        """
        return 'sklearn'

    def fit_on_batch(self, dataset: Dataset) -> None:
        """
        Fits model on batch of data.

        Parameters
        ----------
        dataset: Dataset
          Dataset to train on.
        """

    def get_task_type(self) -> str:
        """
        Returns the task type of the model.
        """

    def get_num_tasks(self) -> int:
        """
        Returns the number of tasks.
        """

    def _fit(self, dataset: Dataset) -> None:
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
        if self.mode is not None and self.mode != dataset.mode:
            raise ValueError(f'The mode of the dataset must match the mode of the model. '
                             f'Got {dataset.mode} for dataset and {self.mode} for model.')
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
        predictions = self.model.predict(dataset.X)

        if len(predictions.shape) > 1:
            if predictions.shape != (len(dataset.mols), dataset.n_tasks):
                predictions = normalize_labels_shape(predictions, dataset.n_tasks)

        return predictions

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        """
        Makes predictions on dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make prediction on.

        Returns
        -------
        np.ndarray
        """
        predictions = self.model.predict_proba(dataset.X)
        if predictions.shape != (len(dataset.mols), dataset.n_tasks):
            predictions = normalize_labels_shape(predictions, dataset.n_tasks)

        if len(predictions.shape) > 1:
            if predictions.shape != (len(dataset.mols), dataset.n_tasks):
                predictions = normalize_labels_shape(predictions, dataset.n_tasks)

        return predictions

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

    def save(self, folder_path: str = None):
        """
        Saves scikit-learn model to disk using joblib, numpy or pickle.
        Supported extensions: .joblib, .pkl, .npy

        Parameters
        ----------
        folder_path: str
            Folder path to save model to.
        """
        if folder_path is None:
            model_path = self.get_model_filename(self.model_dir)
        else:
            if "." in folder_path:
                raise ValueError("folder_path should be a folder, not a file")
            os.makedirs(folder_path, exist_ok=True)
            model_path = self.get_model_filename(folder_path)

        save_to_disk(self.model, model_path)

        # change file path to keep the extension but add _params
        parameters_file_path = model_path.split('.')[0] + '_params.' + model_path.split('.')[1]
        save_to_disk(self.parameters_to_save, parameters_file_path)

    @classmethod
    def load(cls, folder_path: str, **kwargs) -> 'SklearnModel':
        """
        Loads scikit-learn model from joblib or pickle file on disk.
        Supported extensions: .joblib, .pkl

        Parameters
        ----------
        folder_path: str
            Path to model file.

        Returns
        -------
        SklearnModel
            The loaded scikit-learn model.
        """
        if "." in folder_path:
            raise ValueError("model_path should be a folder, not a file")
        model_path = cls.get_model_filename(folder_path)
        model = load_from_disk(model_path)
        # change file path to keep the extension but add _params
        parameters_file_path = model_path.split('.')[0] + '_params.' + model_path.split('.')[1]
        params = load_from_disk(parameters_file_path)
        instance = cls(model=model, model_dir=model_path, **params)
        return instance

    def cross_validate(self,
                       dataset: Dataset,
                       metric: Metric,
                       splitter: Splitter = None,
                       folds: int = 3):
        """
        Performs cross-validation on a dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to perform cross-validation on.
        metric: Metric
            Metric to evaluate model performance.
        splitter: Splitter
            Splitter to use for cross-validation.
        folds: int
            Number of folds to use for cross-validation.

        Returns
        -------
        Tuple[SKlearnModel, float, float, List[float], List[float], float, float]
            The first element is the best model, the second is the train score of the best model, the third is the train
            score of the best model, the fourth is the test scores of all models, the fifth is the average train scores
            of all folds and the sixth is the average test score of all folds.
        """
        if splitter is None:
            splitter = _get_splitter(dataset)

        datasets = splitter.k_fold_split(dataset, folds)

        train_scores = []
        train_score_best_model = 0
        avg_train_score = 0

        test_scores = []
        test_score_best_model = 0
        avg_test_score = 0
        best_model = None
        split = 1

        print("Computing K-fold cross validation")
        for train_ds, test_ds in datasets:
            split += 1
            dummy_model = clone(SklearnModel(model=self.model))

            dummy_model.fit(train_ds)

            train_score = dummy_model.evaluate(train_ds, [metric])[0]
            train_scores.append(train_score[metric.name])
            avg_train_score += train_score[metric.name]

            test_score = dummy_model.evaluate(test_ds, [metric])[0]
            test_scores.append(test_score[metric.name])
            avg_test_score += test_score[metric.name]

            if test_score[metric.name] > test_score_best_model:
                test_score_best_model = test_score[metric.name]
                train_score_best_model = train_score[metric.name]
                best_model = dummy_model

        return best_model, train_score_best_model, test_score_best_model, \
            train_scores, test_scores, avg_train_score / folds, avg_test_score / folds
