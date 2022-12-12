from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
import pandas as pd
from rdkit.Chem import Mol


class Dataset(ABC):
    """
    Abstract base class for datasets
    Subclasses need to implement their own methods based on this class.
    """

    def __init__(self) -> None:
        self._features2keep = None

    def __len__(self) -> int:
        """
        Get the number of elements in the dataset.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def mols(self):
        """
        Get the molecules in the dataset.
        """
        raise NotImplementedError

    @mols.setter
    @abstractmethod
    def mols(self, value: Union[List[str], List[Mol], np.array]):
        """
        Set the molecules in the dataset.
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
    def X(self, value: np.ndarray):
        """
        Set the features in the dataset.
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
    def y(self, value: np.ndarray):
        """
        Set the labels in the dataset.
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
    def ids(self, value: np.ndarray):
        """
        Set the ids in the dataset.
        """
        raise NotImplementedError

    @property
    def features2keep(self):
        """
        Get the features to keep in the dataset.
        """
        return self._features2keep

    @features2keep.setter
    def features2keep(self, value: np.array):
        """
        Set the features to keep in the dataset.
        """
        self._features2keep = value

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
        mols, X, y, ids.
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
    def remove_elements(self, indexes: List[int]):
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
                 mols: Union[np.ndarray, List[str], List[Mol]],
                 X: np.ndarray = None,
                 y: np.ndarray = None,
                 ids: np.ndarray = None,
                 features2keep: np.ndarray = None,
                 n_tasks: int = 1):
        """
        Initialize a NumpyDataset object.

        Parameters
        ----------
        mols: Union[np.ndarray, List[str], List[Mol]]
            The molecules (e.g. SMILES format) vector for this dataset.
            A numpy array of shape `(n_samples,)`.
        X: np.ndarray
            The features array for this dataset.
            A numpy array of arrays of shape (n_samples, features size)
        y: np.ndarray
            The y (tasks) vector for this dataset.
            A numpy array of shape `(n_samples, n_tasks)`.
        ids: np.ndarray
            The ids vector for this dataset.
            A numpy array of shape `(n_samples,)`.
        features2keep: np.ndarray
            The features to keep in the dataset.
        n_tasks: int
            The number of tasks in the dataset.
        """
        super().__init__()
        if not isinstance(mols, np.ndarray):
            mols = np.array(mols)
        if not isinstance(X, np.ndarray) and X is not None:
            X = np.ndarray(X)
        if not isinstance(y, np.ndarray) and y is not None:
            y = np.ndarray(y)
        if not isinstance(ids, np.ndarray) and ids is not None:
            ids = np.ndarray(ids)
        if not isinstance(features2keep, np.ndarray) and features2keep is not None:
            features2keep = np.ndarray(features2keep)

        self.mols = mols

        if features2keep is not None:
            self.features2keep = features2keep
        else:
            self.features2keep = None

        self.X = X
        self.y = y
        self.ids = ids
        self.n_tasks = n_tasks

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        It returns the number of molecules in the dataset.
        """
        return len(self.mols)

    @property
    def mols(self):
        """
        Get the molecules (e.g. SMILES format) vector for this dataset.
        """
        return self._mols

    @mols.setter
    def mols(self, value):
        """
        Set the molecules (e.g. SMILES format) vector for this dataset.
        """
        self._mols = value

    @property
    def n_tasks(self):
        """
        Get the number of tasks in the dataset.
        """
        return self._n_tasks

    @n_tasks.setter
    def n_tasks(self, value):
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
            if self._X.size > 0:
                if self.features2keep is not None:
                    if self.features2keep.size == 0:
                        return np.empty((0, 0))
                    elif len(self._X.shape) == 2:
                        return self._X[:, self.features2keep]
                    else:
                        return self._X

                else:
                    return self._X

            else:
                raise Exception("This dataset has no features")
        else:
            return None

    @X.setter
    def X(self, value: Union[np.array, list, None]):
        """
        Set the features array for this dataset.

        Parameters
        ----------
        value: Union[np.array, list, None]
            The features array for this dataset.
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
    def y(self, value):
        """
        Set the y (tasks) vector for this dataset.

        Parameters
        ----------
        value: np.ndarray
            The y (tasks) vector for this dataset.
        """
        if value is not None and value.size > 0:
            self._y = value
        else:
            self._y = None

    @property
    def ids(self):
        """
        Get the ids vector for this dataset.
        """
        return self._ids

    @ids.setter
    def ids(self, value):
        """
        Set the ids vector for this dataset.

        Parameters
        ----------
        value: np.ndarray
            The ids vector for this dataset.
        """
        if value is not None and value.size > 0:
            self._ids = value
        else:
            self._ids = [i for i in range(self.mols.shape[0])]

    @property
    def features2keep(self):
        """
        Get the features to keep in the dataset.
        """
        return self._features2keep

    @features2keep.setter
    def features2keep(self, value):
        """
        Set the features to keep in the dataset.

        Parameters
        ----------
        value: np.ndarray
            The features to keep in the dataset.
        """
        self._features2keep = value

    def len_mols(self):
        """
        Get the length of the molecules vector.
        """
        return len(self.mols)

    def len_X(self):
        """
        Get the shape of the features' matrix.
        """
        if self.X is not None:
            return self.X.shape
        else:
            return 'X not defined!'

    def len_y(self):
        """
        Get the shape of the y vector.
        """
        if self.y is not None:
            return self.y.shape
        else:
            return 'y not defined!'

    def len_ids(self):
        """
        Get the length of the ids vector.
        """
        if self.ids is not None:
            return self.ids.shape
        else:
            return 'ids not defined!'

    def get_shape(self):
        """
        Get the shape of the dataset.
        Returns four tuples, giving the shape of the mols, X and y arrays.
        """
        print('Mols_shape: ', self.len_mols())
        print('Features_shape: ', self.len_X())
        print('Labels_shape: ', self.len_y())

    def remove_duplicates(self):
        """
        Remove duplicates from the dataset.
        """
        unique, index = np.unique(self.X, return_index=True, axis=0)
        self.select(index, axis=0)

    def remove_elements(self, indexes):
        """
        Remove elements with specific indexes from the dataset
            Very useful when doing feature selection or to remove NAs.
        """
        all_indexes = self.ids
        indexes_to_keep = list(set(all_indexes) - set(indexes))
        self.select(indexes_to_keep)

    def select_features(self, indexes):
        """
        Select features with specific indexes from the dataset
        """
        self.select(indexes, axis=1)

    def remove_nan(self, axis=0):
        """
        Remove only samples with at least one NaN in the features (when axis = 0)
        Or remove samples with all features with NaNs and the features with at least one NaN (axis = 1)
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

                else:
                    if i is None:
                        indexes.append(self.ids[j])
                j += 1
            if len(indexes) > 0:
                print('Elements with indexes: ', indexes, ' were removed due to the presence of NAs!')
                # print('The elements in question are: ', self.mols[indexes])
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

    def select(self, indexes: List[int], axis: int = 0):
        """
        Creates a new sub dataset of self from a selection of indexes.

        Parameters
        ----------
        indexes: List[int]
          List of indices to select.
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

    def merge(self, datasets: List[Dataset]) -> 'NumpyDataset':
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
        flag2 = False

        for ds in datasets:
            mols = np.append(mols, ds.mols, axis=0)
            y = np.append(y, ds.y, axis=0)
            ids = np.append(ids, ds.ids, axis=0)
            if X is not None:
                if len(X[0]) == len(ds.X[0]):
                    X = np.append(X, ds.X, axis=0)
                else:
                    flag2 = False
            else:
                flag2 = False
        if flag2:
            print('Features are not the same length/type... '
                  '\nRecalculate features for all inputs! '
                  '\nAppending empty array in dataset features!')
            return NumpyDataset(mols, None, y, ids)
        else:
            return NumpyDataset(mols, X, y, ids, self.features2keep)

    def save_to_csv(self, path):
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

    # TODO: test load and save
    def load_features(self, path, sep=',', header=0):
        """
        Load features from a csv file.

        Parameters
        ----------
        path: str
            Path to the csv file.
        sep: str
            Delimiter to use.
        header: int
            Row number to use as the column names.
        """
        df = pd.read_csv(path, sep=sep, header=header)
        self.X = df.to_numpy()

    # TODO: Order of the features compared with the initial mols/y's is lost because some features cannot be computed
    #  due to smiles invalidity (use only the function save_to_csv? or think about other implementation?)
    def save_features(self, path='fingerprints.csv'):
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
            raise ValueError('No fingerprint was already calculated!')
