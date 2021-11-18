from abc import ABC, abstractmethod

import joblib

from Datasets.Datasets import Dataset

import numpy as np


class BaseScaler(ABC):

    def __init__(self):
        if self.__class__ == BaseScaler:
            raise Exception('Abstract class MolecularFeaturizer should not be instantiated')

    @abstractmethod
    @property
    def scaler_object(self):
        raise NotImplementedError

    @abstractmethod
    @scaler_object.setter
    def scaler_object(self, value):
        raise NotImplementedError

    def save_scaler(self, file_path):

        joblib.dump(self.scaler_object, file_path)

    @abstractmethod
    def load_scaler(self, file_path: str):
        raise NotImplementedError

    def fit_transform(self, dataset: Dataset):

        try:
            dataset.X = self._fit_transform(dataset.X)

        except:
            raise Exception("It was not possible to scale the data")

    @abstractmethod
    def _fit_transform(self, X):
        raise NotImplementedError

    def fit(self, dataset: Dataset):
        try:
            self._fit(dataset.X)

        except:
            raise Exception("It was not possible to scale the data")

    @abstractmethod
    def _fit(self, X: np.ndarray):
        raise NotImplementedError

    def transform(self, dataset: Dataset):
        try:
            dataset.X = self._transform(dataset.X)

        except:
            raise Exception("It was not possible to scale the data")

    def _transform(self, X: np.ndarray):
        raise NotImplementedError

    #TODO: figure out the better way of wrapping this method, as it intends to fit the dataset in batches
    def partial_fit(self, dataset: Dataset):
        pass
