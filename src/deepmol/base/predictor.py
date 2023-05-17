from abc import abstractmethod
from typing import Union, List, Tuple, Dict

import numpy as np

from deepmol.datasets import Dataset
from deepmol.metrics import Metric


class Predictor:
    """
    Abstract base class for predictors.
    A predictor is an object that can make predictions on a Dataset object.
    All predictors must implement the predict(), predict_proba(), and evaluate() methods.
    """

    def __init__(self):
        """
        Initializes the predictor.
        """
        self.is_fitted_ = False

    @property
    def model_dir(self) -> str:
        """
        Directory where the model will be stored.

        Returns
        -------
        str
            Directory where the model is stored.
        """
        return self.model_dir

    def fit(self, dataset: Dataset) -> 'Predictor':
        """
        Fits a model on data in a Dataset object.

        Parameters
        ----------
        dataset: Dataset
            the Dataset to train on

        Returns
        -------
        Predictor
            self
        """
        self._fit(dataset)
        self.is_fitted_ = True
        return self

    @abstractmethod
    def _fit(self, dataset: Dataset):
        """
        Fits a model on data in a Dataset object.

        Parameters
        ----------
        dataset: Dataset
            the Dataset to train on
        """

    def is_fitted(self) -> bool:
        """
        Whether the predictor is fitted.

        Returns
        -------
        bool
            True if the predictor is fitted, False otherwise.
        """
        return hasattr(self, 'is_fitted_') and self.is_fitted_

    @abstractmethod
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Uses self to make predictions on provided Dataset object.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make prediction on

        Returns
        -------
        np.ndarray
            A numpy array of predictions.
        """

    @abstractmethod
    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        """
        Uses self to make predictions on provided Dataset object.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make prediction on

        Returns
        -------
        np.ndarray
            A numpy array of predictions.
        """

    @abstractmethod
    def evaluate(self,
                 dataset: Dataset,
                 metrics: Union[List[Metric], Metric],
                 per_task_metrics: bool = False) -> Tuple[Dict, Union[None, Dict]]:
        """
        Evaluates the predictor of this model on specified dataset using specified metrics.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        metrics: Union[List[Metric], Metric]
            The set of metrics provided.
        per_task_metrics: bool
            If true, return computed metric for each task on multitask dataset.
        """

    @abstractmethod
    def save(self):
        """
        Saves the predictor to disk.
        """

    @classmethod
    @abstractmethod
    def load(cls, model_dir: str) -> 'Predictor':
        """
        Loads a predictor from disk.

        Parameters
        ----------
        model_dir: str
            Directory where the predictor is stored.

        Returns
        -------
        Predictor
            The loaded predictor.
        """
