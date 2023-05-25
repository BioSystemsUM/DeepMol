from abc import ABC

import joblib

from deepmol.base import Transformer
from deepmol.datasets import Dataset
from deepmol.utils.decorators import modify_object_inplace_decorator


class BaseScaler(ABC, Transformer):
    """
    Abstract class for all scalers. It is used to define the interface for all scalers.
    """

    def __init__(self, scaler, columns: list = None) -> None:
        """
        Constructor for the BaseScaler class.
        """
        if self.__class__ == BaseScaler:
            raise Exception('Abstract class BaseScaler should not be instantiated')
        super().__init__()
        self._scaler_object = scaler
        self.columns = columns

    @property
    def scaler_object(self):
        """
        Returns the scaler object.

        Returns
        -------
        object:
            The scaler object.
        """
        return self._scaler_object

    @scaler_object.setter
    def scaler_object(self, value: object):
        """
        Sets the scaler object.

        Parameters
        ----------
        value: object
            The scaler object.
        """
        self._scaler_object = value

    def save(self, file_path: str) -> None:
        """
        Saves the scaler object to a file.

        file_path: str
            The path to the file where the scaler object will be saved.
        """
        joblib.dump(self._scaler_object, file_path)

    def load(self, file_path: str) -> 'BaseScaler':
        """
        Loads the scaler object from a file.

        file_path: str
            The path to the file where the scaler object is saved.

        Returns
        -------
        object
            The scaler object.
        """
        self._scaler_object = joblib.load(file_path)
        return self

    @modify_object_inplace_decorator
    def scale(self, dataset: Dataset) -> Dataset:
        """
        Scales the dataset.

        dataset: Dataset
            The dataset to be scaled.
        """
        return self.fit_transform(dataset)

    def _fit(self, dataset: Dataset) -> 'BaseScaler':
        """
        Fits the scaler with the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to be fitted.

        Returns
        -------
        BaseScaler
            The fitted scaler.
        """
        if not self.columns:
            self.columns = [i for i in range(dataset.X.shape[1])]
        x = dataset.X[:, self.columns]
        self._scaler_object.fit(x)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset.

        dataset: Dataset
            The dataset to be transformed.
        """
        x = dataset.X[:, self.columns]
        res = self._scaler_object.transform(x)
        dataset.X[:, self.columns] = res
        return dataset
