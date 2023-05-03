from abc import ABC, abstractmethod

import joblib
import numpy as np

from deepmol.datasets import Dataset
from deepmol.utils.decorators import modify_object_inplace_decorator


class BaseScaler(ABC):
    """
    Abstract class for all scalers. It is used to define the interface for all scalers.
    """

    def __init__(self) -> None:
        """
        Constructor for the BaseScaler class.
        """
        if self.__class__ == BaseScaler:
            raise Exception('Abstract class BaseScaler should not be instantiated')

    @property
    @abstractmethod
    def scaler_object(self) -> object:
        """
        Returns the scaler object.

        Returns
        -------
        object
            The scaler object.
        """

    @scaler_object.setter
    @abstractmethod
    def scaler_object(self, value: object) -> None:
        """
        Sets the scaler object.

        value: object
            The scaler object.
        """

    def save(self, file_path: str) -> None:
        """
        Saves the scaler object to a file.

        file_path: str
            The path to the file where the scaler object will be saved.
        """
        joblib.dump(self.scaler_object, file_path)

    @abstractmethod
    def load(self, file_path: str) -> object:
        """
        Loads the scaler object from a file.

        file_path: str
            The path to the file where the scaler object is saved.

        Returns
        -------
        object
            The scaler object.
        """

    @modify_object_inplace_decorator
    def fit_transform(self, dataset: Dataset, columns: list = None) -> Dataset:
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
            return dataset
        except Exception as e:
            raise Exception(f"It was not possible to scale the data. Error: {e}")

    @abstractmethod
    def _fit_transform(self, X: np.ndarray) -> None:
        """
        Fits and transforms the dataset.

        X: np.ndarray
            The dataset to be fitted and transformed.
        """

    @modify_object_inplace_decorator
    def fit(self, dataset: Dataset, columns: list = None) -> Dataset:
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
            return dataset
        except:
            raise Exception("It was not possible to scale the data")

    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        """
        Fits the dataset.

        X: np.ndarray
            The dataset to be fitted.
        """

    @modify_object_inplace_decorator
    def transform(self, dataset: Dataset, columns: list = None) -> Dataset:
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
            dataset.X[:, columns] = res
            return dataset
        except:
            raise Exception("It was not possible to scale the data")

    def _transform(self, X: np.ndarray) -> None:
        """
        Transforms the dataset.

        X: np.ndarray
            The dataset to be transformed.
        """

    # TODO: figure out the better way of wrapping this method, as it intends to fit the dataset in batches
    def partial_fit(self, dataset: Dataset) -> None:
        """
        Partially fits the dataset.

        dataset: Dataset
            The dataset to be partially fitted.
        """
