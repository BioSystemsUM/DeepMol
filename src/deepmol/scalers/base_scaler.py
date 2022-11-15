from abc import ABC, abstractmethod

import joblib
import numpy as np

from deepmol.datasets import Dataset


class BaseScaler(ABC):
    """
    Abstract class for all scalers. It is used to define the interface for all scalers.
    """

    def __init__(self):
        """
        Constructor for the BaseScaler class.
        """
        if self.__class__ == BaseScaler:
            raise Exception('Abstract class BaseScaler should not be instantiated')

    @property
    @abstractmethod
    def scaler_object(self):
        """
        Returns the scaler object.
        """
        raise NotImplementedError

    @scaler_object.setter
    @abstractmethod
    def scaler_object(self, value: object):
        """
        Sets the scaler object.

        value: object
            The scaler object.
        """
        raise NotImplementedError

    def save_scaler(self, file_path: str):
        """
        Saves the scaler object to a file.

        file_path: str
            The path to the file where the scaler object will be saved.
        """
        joblib.dump(self.scaler_object, file_path)

    @abstractmethod
    def load_scaler(self, file_path: str):
        """
        Loads the scaler object from a file.

        file_path: str
            The path to the file where the scaler object is saved.
        """
        raise NotImplementedError

    def fit_transform(self, dataset: Dataset, columns: list = None):
        """
        Fits and transforms the dataset.

        dataset: Dataset
            The dataset to be fitted and transformed.
        columns: list
            The columns to be fitted and transformed.
        """
        if not columns:
            columns = [i for i in range(dataset.X.shape[1])]
        try:
            res = self._fit_transform(dataset.X[:, columns])
            # TODO: due to X being a property, the "set" method must choose so that it could behave as a numpy array
            dataset.X[:, columns] = res

        except:
             raise Exception("It was not possible to scale the data")

    @abstractmethod
    def _fit_transform(self, X: np.ndarray):
        """
        Fits and transforms the dataset.

        X: np.ndarray
            The dataset to be fitted and transformed.
        """
        raise NotImplementedError

    def fit(self, dataset: Dataset, columns: list = None):
        """
        Fits the dataset.

        dataset: Dataset
            The dataset to be fitted.
        columns: list
            The columns to be fitted.
        """
        if not columns:
            columns = [i for i in range(dataset.X.shape[1])]
        try:
            self._fit(dataset.X[:, columns])

        except:
            raise Exception("It was not possible to scale the data")

    @abstractmethod
    def _fit(self, X: np.ndarray):
        """
        Fits the dataset.

        X: np.ndarray
            The dataset to be fitted.
        """
        raise NotImplementedError

    def transform(self, dataset: Dataset, columns: list = None):
        """
        Transforms the dataset.

        dataset: Dataset
            The dataset to be transformed.
        columns: list
            The columns to be transformed.
        """
        if not columns:
            columns = [i for i in range(dataset.X.shape[1])]
        try:
            res = self._transform(dataset.X[:, columns])
            dataset._X[:, columns] = res

        except:
            raise Exception("It was not possible to scale the data")

    def _transform(self, X: np.ndarray):
        """
        Transforms the dataset.

        X: np.ndarray
            The dataset to be transformed.
        """
        raise NotImplementedError

    # TODO: figure out the better way of wrapping this method, as it intends to fit the dataset in batches
    def partial_fit(self, dataset: Dataset):
        """
        Partially fits the dataset.

        dataset: Dataset
            The dataset to be partially fitted.
        """
        raise NotImplementedError
