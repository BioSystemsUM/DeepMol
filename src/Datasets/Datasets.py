from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from typing import Optional, Sequence, Iterable, Union, List

from rdkit.Chem import Mol


class Dataset(ABC):
    """Abstract base class for datasets
    Subclasses need to implement their own methods based on this class.
    """

    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        """Get the number of elements in the dataset."""
        raise NotImplementedError

    @property
    @abstractmethod
    def mols(self):
        raise NotImplementedError

    @mols.setter
    @abstractmethod
    def mols(self, value: Union[List[str], List[Mol]]):
        raise NotImplementedError

    @property
    @abstractmethod
    def X(self):
        raise NotImplementedError

    @X.setter
    @abstractmethod
    def X(self, value: np.ndarray):
        raise NotImplementedError

    @property
    @abstractmethod
    def y(self):
        raise NotImplementedError

    @y.setter
    @abstractmethod
    def y(self, value: np.ndarray):
        raise NotImplementedError

    @property
    @abstractmethod
    def ids(self):
        raise NotImplementedError

    @ids.setter
    @abstractmethod
    def ids(self, value: np.ndarray):
        raise NotImplementedError

    @property
    @abstractmethod
    def n_tasks(self):
        raise NotImplementedError

    @n_tasks.setter
    @abstractmethod
    def n_tasks(self, value: int):
        raise NotImplementedError

    @abstractmethod
    def get_shape(self):
        """Get the shape of all the elements of the dataset.
        mols, X, y, ids.
        """
        raise NotImplementedError

    @abstractmethod
    def get_mols(self) -> np.ndarray:
        """Get the molecules (e.g. SMILES format) vector for this dataset as a single numpy array."""
        raise NotImplementedError

    @abstractmethod
    def get_X(self) -> np.ndarray:
        """Get the features array for this dataset as a single numpy array."""
        raise NotImplementedError

    @abstractmethod
    def get_y(self) -> np.ndarray:
        """Get the y (tasks) vector for this dataset as a single numpy array."""
        raise NotImplementedError

    @abstractmethod
    def get_ids(self) -> np.ndarray:
        """Get the ids vector for this dataset as a single numpy array."""
        raise NotImplementedError

    @abstractmethod
    def remove_nan(self, axis: int = 0):
        raise NotImplementedError

    @abstractmethod
    def remove_elements(self, indexes: List[int]):
        raise NotImplementedError

    @abstractmethod
    def select_features(self, indexes: List[int]):
        raise NotImplementedError

    @abstractmethod
    def select(self, indexes: List[int], axis: int = 0):
        raise NotImplementedError

    @abstractmethod
    def select_to_split(self, indexes: List[int]):
        raise NotImplementedError


class NumpyDataset(Dataset):
    """A Dataset defined by in-memory numpy arrays.
      This subclass of 'Dataset' stores arrays mols, X, y, ids in memory as
      numpy arrays.
      """

    @property
    def mols(self):
        return self._mols

    @mols.setter
    def mols(self, value):
        self._mols = value

    @property
    def n_tasks(self):
        return self._n_tasks

    @n_tasks.setter
    def n_tasks(self, value):
        self._n_tasks = value

    @property
    def X(self):
        if self._X is not None:
            if self._X.size > 0:
                if self.features2keep is not None:
                    if self.features2keep.size == 0:
                        raise Exception("This dataset has no features")
                    else:
                        return self._X[:, self.features2keep]

                else:
                    return self._X

            else:
                raise Exception("This dataset has no features")
        else:
            return None

    @X.setter
    def X(self, value: Union[np.array, None]):
        if value is not None and value.size > 0:
            self.features2keep = np.array([i for i in range(value.shape[1])])
            self._X = value
        else:
            self._X = None

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        if value is not None and value.size > 0:
            self._y = value
        else:
            self._y = None

    @property
    def ids(self):
        return self._ids

    @ids.setter
    def ids(self, value):
        if value is not None and value.size > 0:
            self._ids = value
        else:
            self._ids = None

    def __len__(self) -> int:
        return len(self.mols)

    def __init__(self, mols: Union[np.ndarray, List[str], List[Mol]], X: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 ids: Optional[np.ndarray] = None, features2keep: Optional[np.ndarray] = None, n_tasks: int = 1):
        """Initialize a NumpyDataset object.
        Parameters
        ----------
        mols: np.ndarray
          Input features. A numpy array of shape `(n_samples,)`.
        X: np.ndarray, optional (default None)
          Features. A numpy array of arrays of shape (n_samples, features size)
        y: np.ndarray, optional (default None)
          Labels. A numpy array of shape `(n_samples,)`. Note that each label can
          have an arbitrary shape.
        ids: np.ndarray, optional (default None)
          Identifiers. A numpy array of shape (n_samples,)
        features2keep: np.ndarray, optional (deafult None)
          Indexes of the features of X to keep.
        n_tasks: int, default 1
          Number of learning tasks.
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
        self.X = X
        self.y = y
        self.ids = ids
        self.features2keep = features2keep
        self.n_tasks = n_tasks

        if features2keep is not None:
            print(self.len_X())
            if len(features2keep) != self.len_X()[1]:
                try:
                    self.select_features(features2keep)
                    print('Defined features extracted!')
                except Exception as e:
                    print('Error while removing defined features!')
                    print(e)

    def len_mols(self):
        return len(self.mols)

    def len_X(self):
        if self.X is not None:
            return self.X.shape
        else:
            return 'X not defined!'

    def len_y(self):
        if self.y is not None:
            return self.y.shape
        else:
            return 'y not defined!'

    def len_ids(self):
        if self.ids is not None:
            return self.ids.shape
        else:
            return 'ids not defined!'

    def get_shape(self):
        """Get the shape of the dataset.
        Returns four tuples, giving the shape of the mols, X and y arrays.
        """
        print('Mols_shape: ', self.len_mols())
        print('Features_shape: ', self.len_X())
        print('Labels_shape: ', self.len_y())

    def get_mols(self) -> Union[List[str], List[Mol], None]:
        """Get the features array for this dataset as a single numpy array."""
        if self.mols is not None:
            return self.mols
        else:
            print("Molecules not defined!")
            return None

    def get_X(self) -> Union[np.ndarray, None]:
        """Get the X vector for this dataset as a single numpy array."""
        if self.X is not None:
            return self.X
        else:
            print("X not defined!")
            return None

    def get_y(self) -> Union[np.ndarray, None]:
        """Get the y vector for this dataset as a single numpy array."""
        if self.y is not None:
            return self.y
        else:
            print("y not defined!")
            return None

    def get_ids(self) -> Union[np.ndarray, None]:
        """Get the ids vector for this dataset as a single numpy array."""
        if self.ids is not None:
            return self.ids
        else:
            print("ids not defined!")
            return None

    def remove_duplicates(self):
        unique, index = np.unique(self.X, return_index=True, axis=0)

        self.select(index, axis=0)

    def remove_elements(self, indexes):
        """Remove elements with specific indexes from the dataset
            Very useful when doing feature selection or to remove NAs.
        """
        all_indexes = self.ids
        indexes_to_keep = list(set(all_indexes) - set(indexes))

        self.select(indexes_to_keep)

    def select_features(self, indexes):
        self.select(indexes, axis=1)

    def remove_nan(self, axis=0):
        """Remove only samples with at least one NaN in the features (when axis = 0)
           Or remove samples with all features with NaNs and the features with at least one NaN (axis = 1) """

        j = 0
        indexes = []

        if axis == 0:
            for i in self.X:
                if np.isnan(np.dot(i, i)):
                    indexes.append(j)

                j += 1
            if len(indexes) > 0:
                print('Elements with indexes: ', indexes, ' were removed due to the presence of NAs!')
                print('The elements in question are: ', self.mols[indexes])
                self.remove_elements(indexes)

        elif axis == 1:
            self.X = self.X[~np.isnan(self.X).all(axis=1)]
            nans_column_indexes = [nans_indexes[1] for nans_indexes in np.argwhere(np.isnan(self.X))]

            column_sets = list(set(nans_column_indexes))
            self.X = np.delete(self.X, column_sets, axis=1)

    def select_to_split(self, indexes: List[int]):

        y = None
        X = None
        ids = None

        mols = [self.mols[i] for i in indexes]

        if self.y is not None:
            y = self.y[indexes]

        if self.X is not None:
            X = self.X[indexes, :]

        if self.ids is not None:
            ids = self.ids[indexes]

        return NumpyDataset(mols, X, y, ids, self.features2keep)

    def select(self, indexes: Sequence[int], axis: int = 0):
        """Creates a new subdataset of self from a selection of indexes.
        Parameters
        ----------
        indexes: List[int]
          List of indices to select.
        axis: int
          Axis

        Returns
        -------
        Dataset
          A NumpyDataset object containing only the selected indexes.
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

    def merge(self,
              datasets: Iterable[Dataset]) -> 'NumpyDataset':
        """Merges provided datasets with the self dataset.
        Parameters
        ----------
        datasets: Iterable[Dataset]
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
            return NumpyDataset(mols, np.empty(), y, ids)
        else:
            return NumpyDataset(mols, X, y, ids, self.features2keep)

    def save_to_csv(self, path):
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
        df = pd.read_csv(path, sep=sep, header=header)
        self.X = df.to_numpy()

    # TODO: Order of the features compared with the initial mols/y's is lost because some features cannot be computed
    #  due to smiles invalidity (use only the function save_to_csv? or think about other implementation?)
    def save_features(self, path='fingerprints.csv'):
        if self.X is not None:
            columns_names = ['feat_' + str(i + 1) for i in range(self.X.shape[1])]
            df = pd.DataFrame(self.X, columns=columns_names)
            df.to_csv(path, index=False)
        else:
            raise ValueError('No fingerprint was already calculated!')


'''
#TODO: implement a Dataset subclass to use/deal with datasets in disk instead of in-memory
class DiskDataset(Dataset):
    """
    ...
    """

    def __init__(self) -> None:
        raise NotImplementedError()

'''
