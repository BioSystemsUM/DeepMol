from abc import ABC, abstractmethod

import joblib

from datasets.datasets import Dataset

import numpy as np


class BaseScaler(ABC):

    def __init__(self):
        if self.__class__ == BaseScaler:
            raise Exception('Abstract class BaseScaler should not be instantiated')

    @property
    @abstractmethod
    def scaler_object(self):
        raise NotImplementedError

    @scaler_object.setter
    @abstractmethod
    def scaler_object(self, value):
        raise NotImplementedError

    def save_scaler(self, file_path):

        joblib.dump(self.scaler_object, file_path)

    @abstractmethod
    def load_scaler(self, file_path: str):
        raise NotImplementedError

    def fit_transform(self, dataset: Dataset, columns=None):
        if not columns:
            columns = [i for i in range(dataset.X.shape[1])]
        try:
            res = self._fit_transform(dataset.X[:, columns])
            # TODO: due to X being a property, the "set" method must choose so that it could behave as a numpy array
            dataset.X[:, columns] = res

        except:
             raise Exception("It was not possible to scale the data")

    @abstractmethod
    def _fit_transform(self, X):
        raise NotImplementedError

    def fit(self, dataset: Dataset, columns=None):
        if not columns:
            columns = [i for i in range(dataset.X.shape[1])]
        try:
            self._fit(dataset.X[:, columns])

        except:
            raise Exception("It was not possible to scale the data")

    @abstractmethod
    def _fit(self, X: np.ndarray):
        raise NotImplementedError

    def transform(self, dataset: Dataset, columns=None):
        if not columns:
            columns = [i for i in range(dataset.X.shape[1])]
        try:
            res = self._transform(dataset.X[:, columns])
            dataset._X[:, columns] = res

        except:
            raise Exception("It was not possible to scale the data")

    def _transform(self, X: np.ndarray):
        raise NotImplementedError

    # TODO: figure out the better way of wrapping this method, as it intends to fit the dataset in batches
    def partial_fit(self, dataset: Dataset):
        pass
