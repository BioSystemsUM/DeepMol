import uuid
from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
import pandas as pd

from deepmol.datasets._utils import merge_arrays, merge_arrays_of_arrays


class Dataset(ABC):
    """
    Abstract base class for datasets
    Subclasses need to implement their own methods based on this class.
    """

    @property
    @abstractmethod
    def mols(self):
        """
        Get the molecules in the dataset.
        """
        raise NotImplementedError

    @mols.setter
    @abstractmethod
    def mols(self, value: Union[List[str], np.array]):
        """
        Set the molecules in the dataset.

        Parameters
        ----------
        value: Union[List[str], np.array]
            The molecules to set in the dataset.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def X(self):
        """
        Get the features in the dataset.
        """
        raise NotImplementedError

    @X.setter
    @abstractmethod
    def X(self, value: Union[List, np.array]):
        """
        Set the features in the dataset.

        Parameters
        ----------
        value: Union[List, np.array]
            The features to set in the dataset.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def y(self):
        """
        Get the labels in the dataset.
        """
        raise NotImplementedError

    @y.setter
    @abstractmethod
    def y(self, value: Union[List, np.array]):
        """
        Set the labels in the dataset.

        Parameters
        ----------
        value: Union[List, np.array]
            The labels to set in the dataset.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def ids(self):
        """
        Get the ids in the dataset.
        """
        raise NotImplementedError

    @ids.setter
    @abstractmethod
    def ids(self, value: Union[List, np.array]):
        """
        Set the ids in the dataset.

        Parameters
        ----------
        value: Union[List[str], np.array]
            The ids to set in the dataset.
        """
        raise NotImplementedError

    @property
    def features2keep(self):
        """
        Get the features to keep in the dataset.
        """
        raise NotImplementedError

    @features2keep.setter
    def features2keep(self, value: Union[List, np.array]):
        """
        Set the features to keep in the dataset.

        Parameters
        ----------
        value: Union[List, np.array]
            The features to keep in the dataset.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_tasks(self):
        """
        Get the number of tasks in the dataset.
        """
        raise NotImplementedError

    @n_tasks.setter
    @abstractmethod
    def n_tasks(self, value: int):
        """
        Set the number of tasks in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def get_shape(self):
        """
        Get the shape of all the elements of the dataset.
        mols, X, y.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_nan(self, axis: int = 0):
        """
        Remove the nan values from the dataset.

        Parameters
        ----------
        axis: int
            The axis to remove the nan values.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_elements(self, indexes: List):
        """
        Remove the elements from the dataset.

        Parameters
        ----------
        indexes: List[int]
            The indexes of the elements to remove.
        """
        raise NotImplementedError

    @abstractmethod
    def select_features(self, indexes: List[int]):
        """
        Select the features from the dataset.

        Parameters
        ----------
        indexes: List[int]
            The indexes of the features to select.
        """
        raise NotImplementedError

    @abstractmethod
    def select(self, indexes: List[int], axis: int = 0):
        """
        Select the elements from the dataset.

        Parameters
        ----------
        indexes: List[int]
            The indexes of the elements to select.
        axis: int
            The axis to select the elements.
        """
        raise NotImplementedError

    @abstractmethod
    def select_to_split(self, indexes: List[int]):
        """
        Select the elements from the dataset to split.

        Parameters
        ----------
        indexes: List[int]
            The indexes of the elements to select.
        """
        raise NotImplementedError


class NumpyDataset(Dataset):
    """
    A Dataset defined by in-memory numpy arrays.
    This subclass of 'Dataset' stores arrays mols, X, y, ids in memory as numpy arrays.
    """

    def __init__(self,
                 mols: Union[np.array, List[str]],
                 X: Union[List, np.array] = None,
                 y: Union[List, np.array] = None,
                 ids: Union[List, np.array] = None,
                 features2keep: Union[List, np.array] = None,
                 n_tasks: int = 1):
        """
        Initialize a NumpyDataset object.

        Parameters
        ----------
        mols: Union[np.array, List[str]]
            The molecules in the dataset.
        X: Union[List, np.array]
            The features in the dataset.
        y: Union[List, np.array]
            The labels in the dataset.
        ids: Union[List[int], np.array]
            The ids in the dataset. ids will be treated as strings.
        features2keep: Union[List, np.array]
            The features to keep in the dataset.
        n_tasks: int
            The number of tasks in the dataset.
        """
        super().__init__()
        if not isinstance(mols, np.ndarray):
            mols = np.array(mols)
        if not isinstance(X, np.ndarray) and X is not None:
            X = np.array(X)
        if not isinstance(y, np.ndarray) and y is not None:
            y = np.array(y)
        if not isinstance(ids, np.ndarray) and ids is not None:
            ids = np.array([str(i) for i in ids])
        if not isinstance(features2keep, np.ndarray) and features2keep is not None:
            features2keep = np.array(features2keep)

        self._mols = mols

        if features2keep is not None:
            self._features2keep = features2keep
        elif X is not None:
            if len(X.shape) == 1:
                self._features2keep = np.arange(X.shape[0])
            else:
                self._features2keep = np.arange(X.shape[1])
        else:
            self._features2keep = None

        self._X = X
        self._y = y
        if ids is None:
            ids = np.array([str(uuid.uuid4().hex) for i in range(len(mols))])
        self._ids = ids
        self._n_tasks = n_tasks

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        It returns the number of molecules in the dataset.
        """
        return len(self._mols)

    @property
    def mols(self):
        """
        Get the molecules (e.g. SMILES format) vector for this dataset.
        """
        return self._mols

    @mols.setter
    def mols(self, value: Union[np.array, List[str]]):
        """
        Set the molecules (e.g. SMILES format) vector for this dataset.

        Parameters
        ----------
        value: Union[np.array, List[str]]
            The molecules (SMILES format) vector for this dataset.
        """
        self._mols = np.array(value)

    @property
    def n_tasks(self):
        """
        Get the number of tasks in the dataset.
        """
        return self._n_tasks

    @n_tasks.setter
    def n_tasks(self, value: int):
        """
        Set the number of tasks in the dataset.

        Parameters
        ----------
        value: int
            The number of tasks in the dataset.
        """
        self._n_tasks = value

    @property
    def X(self):
        """
        Get the features array for this dataset.
        """
        if self._X is not None:
            if self.features2keep.size == 0:
                return np.empty((0, 0))
            elif len(self._X.shape) == 2:
                return self._X[:, self.features2keep]
            else:
                return self._X
        else:
            return None

    @X.setter
    def X(self, value: Union[np.array, List]):
        """
        Set the features array for this dataset.

        Parameters
        ----------
        value: Union[np.array, List]
            The features for this dataset.
        """
        if isinstance(value, list):
            value = np.array(value)
        if value is not None and value.size > 0:
            if len(value.shape) == 2:
                self.features2keep = np.array([i for i in range(value.shape[1])])
            else:
                self.features2keep = np.array([i for i in range(len(value))])
            self._X = value
        else:
            self._X = None

    @property
    def y(self):
        """
        Get the y (tasks) vector for this dataset.
        """
        return self._y

    @y.setter
    def y(self, value: Union[np.array, List]):
        """
        Set the y (tasks) vector for this dataset.

        Parameters
        ----------
        value: Union[np.array, List]
            The y (tasks) vector for this dataset.
        """
        if len(value) != len(self.mols):
            raise ValueError("Length of y vector must be equal to length of mols vector")
        if isinstance(value, list):
            value = np.array(value)
        self._y = value

    @property
    def ids(self):
        """
        Get the ids vector for this dataset.
        """
        return self._ids

    @ids.setter
    def ids(self, value: Union[np.array, List]):
        """
        Set the ids vector for this dataset. ids will be treated as strings.

        Parameters
        ----------
        value: Union[np.array, List]
            The ids vector for this dataset.
        """
        if value is None:
            self._ids = [str(uuid.uuid4().hex) for i in range(self.mols.shape[0])]
        elif len(set(value)) != len(value):
            raise ValueError(f"Ids must be unique! Got {value}.")
        elif len(value) != len(self.mols):
            raise ValueError(f"Length of ids vector must be equal to length of mols vector. "
                             f"Got {len(value)} values and {len(self.mols)} molecules.")
        else:
            self._ids = np.array([str(i) for i in value])

    @property
    def features2keep(self):
        """
        Get the features to keep in the dataset.
        """
        return self._features2keep

    @features2keep.setter
    def features2keep(self, value: Union[np.array, List]):
        """
        Set the features to keep in the dataset.

        Parameters
        ----------
        value: Union[np.array, List]
            The features to keep in the dataset.
        """
        self._features2keep = np.array(value)

    def get_shape(self):
        """
        Get the shape of the dataset.
        Returns four tuples, giving the shape of the mols, X and y arrays.
        """
        print(f'Mols_shape: {self.mols.shape}')
        if self.X is not None:
            x_shape = self.X.shape
            print(f'Features_shape: {x_shape}')
        else:
            x_shape = None
            print(f'Features_shape: {None}')
        if self.y is not None:
            y_shape = self.y.shape
            print(f'Labels_shape: {y_shape}')
        else:
            y_shape = None
            print(f'Labels_shape: {None}')
        return self.mols.shape, x_shape, y_shape

    def remove_duplicates(self):
        """
        Remove duplicated features from the dataset.
        """
        unique, index = np.unique(self.X, return_index=True, axis=0)
        ids = self.ids[index]
        self.select(ids, axis=0)

    def remove_elements(self, indexes: Union[List[str], List[int]]):
        """
        Remove elements with specific IDs from the dataset.

        Parameters
        ----------
        indexes: Union[List[str], List[int]]
            The IDs of the elements to remove from the dataset.
            IDs can be either strings or integers (not both).
        """
        indexes = [str(i) for i in indexes]
        all_indexes = self.ids
        indexes_to_keep = list(set(all_indexes) - set(indexes))
        self.select(indexes_to_keep)

    def select_features(self, indexes: List[int]):
        """
        Select features with specific indexes from the dataset

        Parameters
        ----------
        indexes: List[int]
            The indexes of the features to select from the dataset.
        """
        self.select(indexes, axis=1)

    def remove_nan(self, axis: int = 0):
        """
        Remove only samples with at least one NaN in the features (when axis = 0)
        Or remove samples with all features with NaNs and the features with at least one NaN (axis = 1)

        Parameters
        ----------
        axis: int
            The axis to remove the NaNs from.
        """
        j = 0
        indexes = []

        if axis == 0:
            shape = self.X.shape
            X = self.X
            for i in X:
                if len(shape) == 2:
                    if np.isnan(np.dot(i, i)):
                        indexes.append(self.ids[j])
                elif isinstance(i, float) or isinstance(i, int):
                    if i is None or np.isnan(i):
                        indexes.append(self.ids[j])
                j += 1
            if len(indexes) > 0:
                print('Elements with IDs: ', indexes, ' were removed due to the presence of NAs!')
                self.remove_elements(indexes)

        elif axis == 1:
            self.X = self.X[~np.isnan(self.X).all(axis=1)]
            nans_column_indexes = [nans_indexes[1] for nans_indexes in np.argwhere(np.isnan(self.X))]

            column_sets = list(set(nans_column_indexes))
            self.X = np.delete(self.X, column_sets, axis=1)

    def select_to_split(self, indexes: List[int]):
        """
        Select elements with specific indexes to split the dataset

        Parameters
        ----------
        indexes: List[int]
            The indexes of the elements to split the dataset.
        """
        y = None
        X = None
        ids = None

        mols = [self.mols[i] for i in indexes]

        if self.y is not None:
            y = self.y[indexes]

        if self.X is not None:
            if len(self.X.shape) == 2:
                X = self.X[indexes, :]
            else:
                X = self.X[indexes]

        if self.ids is not None:
            ids = self.ids[indexes]
        return NumpyDataset(mols, X, y, ids, self.features2keep)

    def select(self, indexes: Union[List[str], List[int]], axis: int = 0):
        """
        Creates a new sub dataset of self from a selection of indexes.

        Parameters
        ----------
        indexes: Union[List[str], List[int]]
          List of ids/indexes to select.
          IDs in case axis = 0, indexes in case axis = 1.
        axis: int
            Axis to select along. 0 selects along the first axis, 1 selects along the second axis.
        """

        if axis == 0:
            all_indexes = self.ids
            indexes_to_delete = sorted(list(set(all_indexes) - set(indexes)))
            raw_indexes = []
            for index in indexes_to_delete:
                for i, mol_index in enumerate(all_indexes):
                    if index == mol_index:
                        raw_indexes.append(i)

            self.mols = np.delete(self.mols, raw_indexes, axis)

            if self.y is not None:
                self.y = np.delete(self.y, raw_indexes, axis)

            if self.X is not None:
                self.X = np.delete(self.X, raw_indexes, axis)

            if self.ids is not None:
                self.ids = np.delete(self.ids, raw_indexes, axis)

        if axis == 1:
            indexes_to_delete = list(set(self.features2keep) - set(indexes))
            self.features2keep = np.array(list(set(self.features2keep) - set(indexes_to_delete)))
            self.features2keep = np.sort(self.features2keep)

    def merge(self, datasets: List[Dataset]):
        """
        Merges provided datasets with the self dataset.

        Parameters
        ----------
        datasets: List[Dataset]
            List of datasets to merge.

        Returns
        -------
        NumpyDataset
            A merged NumpyDataset.
        """
        datasets = list(datasets)

        X = self.X
        y = self.y
        ids = self.ids
        mols = self.mols

        for ds in datasets:
            ids = merge_arrays(ids, len(mols), ds.ids, len(ds.mols))
            if len(set(ids)) != len(ids):
                raise ValueError(f'IDs must be unique! IDs are {ids}')
            y = merge_arrays(y, len(mols), ds.y, len(ds.mols))
            if X is None or ds.X is None:
                print('Features are not the same length/type... Recalculate features for all inputs!')
                X = None
            elif len(X.shape) == 1 and len(ds.X.shape) == 1:
                X = merge_arrays(X, len(mols), ds.X, len(ds.mols))
            else:
                X = merge_arrays_of_arrays(X, ds.X)
            mols = np.append(mols, ds.mols, axis=0)
        return NumpyDataset(mols, X, y, ids, self.features2keep)

    def to_csv(self, path: str):
        """
        Save the dataset to a csv file.

        Parameters
        ----------
        path: str
            Path to save the csv file.
        """
        df = pd.DataFrame()
        if self.ids is not None:
            df['ids'] = pd.Series(self.ids)
        df['mols'] = pd.Series(self.mols)
        if self.y is not None:
            df['y'] = pd.Series(self.y)
        if self.X is not None:
            columns_names = ['feat_' + str(i + 1) for i in range(self.X.shape[1])]
            df_x = pd.DataFrame(self.X, columns=columns_names)
            df = pd.concat([df, df_x], axis=1)

        df.to_csv(path, index=False)

    def load_features(self, path: str, **kwargs):
        """
        Load features from a csv file.

        Parameters
        ----------
        path: str
            Path to the csv file.
        kwargs:
            Keyword arguments to pass to pandas.read_csv.
        """
        df = pd.read_csv(path, **kwargs)
        self.X = df.to_numpy()

    def save_features(self, path: str = 'features.csv'):
        """
        Save the features to a csv file.

        Parameters
        ----------
        path: str
            Path to save the csv file.
        """
        if self.X is not None:
            columns_names = ['feat_' + str(i + 1) for i in range(self.X.shape[1])]
            df = pd.DataFrame(self.X, columns=columns_names)
            df.to_csv(path, index=False)
        else:
            raise ValueError('Features array is empty!')
