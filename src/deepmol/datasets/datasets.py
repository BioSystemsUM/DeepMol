import uuid
import warnings
from abc import ABC, abstractmethod
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from rdkit.Chem import Mol

from deepmol.loggers.logger import Logger
from deepmol.datasets._utils import merge_arrays, merge_arrays_of_arrays
from deepmol.utils.utils import smiles_to_mol, mol_to_smiles


class Dataset(ABC):
    """
    Abstract base class for datasets
    Subclasses need to implement their own methods based on this class.
    """

    def __init__(self):
        self.logger = Logger()

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        It returns the number of molecules in the dataset.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def smiles(self) -> np.ndarray:
        """
        Get the smiles in the dataset.
        Returns
        -------
        mols : np.ndarray
            Molecule smiles in the dataset.
        """
        raise NotImplementedError

    @smiles.setter
    @abstractmethod
    def smiles(self, value: Union[List[str], np.ndarray]) -> None:
        """
        Set the molecules in the dataset.
        Parameters
        ----------
        value: Union[List[str], np.ndarray]
            The molecules to set in the dataset.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def mols(self) -> np.ndarray:
        """
        Get the molecules in the dataset.

        Returns
        -------
        mols : np.ndarray
            Molecules in the dataset.
        """
        raise NotImplementedError

    @mols.setter
    @abstractmethod
    def mols(self, value: Union[List[str], np.ndarray]) -> None:
        """
        Set the molecules in the dataset.

        Parameters
        ----------
        value: Union[List[str], np.ndarray]
            The molecules to set in the dataset.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def X(self) -> np.ndarray:
        """
        Get the features in the dataset.

        Returns
        -------
        X: np.ndarray
            The features in the dataset.
        """
        raise NotImplementedError

    @X.setter
    @abstractmethod
    def X(self, value: Union[List, np.ndarray]) -> None:
        """
        Set the features in the dataset.

        Parameters
        ----------
        value: Union[List, np.ndarray]
            The features to set in the dataset.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def y(self) -> np.ndarray:
        """
        Get the labels in the dataset.

        Returns
        -------
        y: np.ndarray
            The labels in the dataset.
        """
        raise NotImplementedError

    @y.setter
    @abstractmethod
    def y(self, value: Union[List, np.ndarray]) -> None:
        """
        Set the labels in the dataset.

        Parameters
        ----------
        value: Union[List, np.ndarray]
            The labels to set in the dataset.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def ids(self) -> np.ndarray:
        """
        Get the ids in the dataset.

        Returns
        -------
        ids: np.ndarray
            The ids in the dataset.
        """
        raise NotImplementedError

    @ids.setter
    @abstractmethod
    def ids(self, value: Union[List, np.ndarray]) -> None:
        """
        Set the ids in the dataset.

        Parameters
        ----------
        value: Union[List[str], np.ndarray]
            The ids to set in the dataset.
        """
        raise NotImplementedError

    @property
    def feature_names(self) -> np.ndarray:
        """
        Get the feature labels of the molecules in the dataset.

        Returns
        -------
        feature_names: np.ndarray
            Feature names of the molecules.
        """
        raise NotImplementedError

    @feature_names.setter
    def feature_names(self, value: Union[List, np.ndarray]) -> None:
        """
        Set the feature labels of the molecules in the dataset.

        Parameters
        ----------
        value: Union[List, np.ndarray]
            Feature names of the molecules.
        """
        raise NotImplementedError

    @property
    def label_names(self) -> np.ndarray:
        """
        Get the labels names of the molecules in the dataset.

        Returns
        -------
        label_names: np.ndarray
            Label names of the molecules.
        """
        raise NotImplementedError

    @label_names.setter
    def label_names(self, value: Union[List, np.ndarray]) -> None:
        """
        Set the labels names of the molecules in the dataset.

        Parameters
        ----------
        value: Union[List, np.ndarray]
            Label names of the molecules.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_tasks(self) -> int:
        """
        Get the number of tasks in the dataset.

        Returns
        -------
        n_tasks: int
            The number of tasks in the dataset.
        """
        raise NotImplementedError

    @n_tasks.setter
    @abstractmethod
    def n_tasks(self, value: int) -> None:
        """
        Set the number of tasks in the dataset.

        Parameters
        ----------
        value: int
            The number of tasks in the dataset.

        """
        raise NotImplementedError

    @abstractmethod
    def get_shape(self) -> tuple:
        """
        Get the shape of molecules, features and labels in the dataset.

        Returns
        -------
        shape: tuple
            The shape of molecules, features and labels.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_nan(self, axis: int = 0) -> None:
        """
        Remove the nan values from the dataset.

        Parameters
        ----------
        axis: int
            The axis to remove the nan values.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_elements(self, indexes: List) -> None:
        """
        Remove the elements from the dataset.

        Parameters
        ----------
        indexes: List[int]
            The indexes of the elements to remove.
        """
        raise NotImplementedError

    @abstractmethod
    def select_features_by_index(self, indexes: List[int]) -> None:
        """
        Select the features from the dataset.
        Parameters
        ----------
        indexes: List[int]
            The indexes of the features to select.
        """
        raise NotImplementedError

    @abstractmethod
    def select_features_by_name(self, names: List[str]) -> None:
        """
        Select features with specific names from the dataset
        Parameters
        ----------
        names: List[str]
            The names of the features to select from the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def select(self, indexes: List[int], axis: int = 0) -> None:
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
    def select_to_split(self, indexes: Union[np.ndarray, List[int]]) -> 'Dataset':
        """
        Select the elements from the dataset to split.

        Parameters
        ----------
        indexes: Union[np.ndarray, List[int]]
            The indexes of the elements to select.
        """
        raise NotImplementedError


class SmilesDataset(Dataset):
    """
    A Dataset defined by in-memory numpy arrays.
    This subclass of 'Dataset' stores arrays for smiles strings, Mol objects, features X, labels y, and molecule ids in
    memory as numpy arrays.
    """

    def __init__(self,
                 smiles: Union[np.ndarray, List[str]],
                 mols: Union[np.ndarray, List[Mol]] = None,
                 ids: Union[List, np.ndarray] = None,
                 X: Union[List, np.ndarray] = None,
                 feature_names: Union[List, np.ndarray] = None,
                 y: Union[List, np.ndarray] = None,
                 label_names: Union[List, np.ndarray] = None) -> None:
        """
        Initialize a dataset from SMILES strings.
        Parameters
        ----------
        smiles: Union[np.ndarray, List[str]]
            SMILES strings of the molecules.
        mols: Union[np.ndarray, List[Mol]]
            RDKit Mol objects of the molecules.
        ids: Union[List, np.ndarray]
            IDs of the molecules.
        X: Union[List, np.ndarray]
            Features of the molecules.
        feature_names: Union[List, np.ndarray]
            Names of the features.
        y: Union[List, np.ndarray]
            Labels of the molecules.
        label_names: Union[List, np.ndarray]
            Names of the labels.
        """
        super().__init__()
        self._smiles = np.array(smiles)
        self._ids = np.array([str(i) for i in ids]) if ids is not None \
            else np.array([str(uuid.uuid4().hex) for _ in range(len(smiles))])
        self._X = np.array(X) if X is not None else None
        self._y = np.array(y) if y is not None else None
        self._mols = np.array(mols) if mols is not None else np.array([smiles_to_mol(s) for s in self._smiles])
        self.remove_elements([self._ids[i] for i, m in enumerate(self._mols) if m is None])
        self._feature_names = np.array(feature_names) if feature_names is not None else None
        self._label_names = np.array(label_names) if label_names is not None else None
        self._validate_params()
        self._n_tasks = len(self._label_names) if self._label_names is not None else 0

    @classmethod
    def from_mols(cls,
                  mols: Union[np.ndarray, List[Mol]],
                  ids: Union[List, np.ndarray] = None,
                  X: Union[List, np.ndarray] = None,
                  feature_names: Union[List, np.ndarray] = None,
                  y: Union[List, np.ndarray] = None,
                  label_names: Union[List, np.ndarray] = None) -> 'SmilesDataset':
        """
        Initialize a dataset from RDKit Mol objects.

        Parameters
        ----------
        mols: Union[np.ndarray, List[Mol]]
            RDKit Mol objects of the molecules.
        ids: Union[List, np.ndarray]
            IDs of the molecules.
        X: Union[List, np.ndarray]
            Features of the molecules.
        feature_names: Union[List, np.ndarray]
            Names of the features.
        y: Union[List, np.ndarray]
            Labels of the molecules.
        label_names: Union[List, np.ndarray]
            Names of the labels.

        Returns
        -------
        SmilesDataset
            The dataset instance.
        """
        smiles = np.array([mol_to_smiles(m) for m in mols])
        return cls(smiles, mols, ids, X, feature_names, y, label_names)

    def __len__(self) -> int:
        """
        Get the number of molecules in the dataset.
        Returns
        -------
        int
            Number of molecules in the dataset.
        """
        return len(self._smiles)

    def _validate_params(self) -> None:
        """
        Validates the parameters of the dataset.
        """
        if len(self._smiles) != len(self._ids):
            raise ValueError('Length of smiles and ids must be the same.')
        if self._X is not None and len(self._smiles) != len(self._X):
            raise ValueError('Length of smiles and X must be the same.')
        if self._y is not None and len(self._smiles) != len(self._y):
            raise ValueError('Length of smiles and y must be the same.')
        if self._feature_names is not None and self._X is not None:
            if len(self._X.shape) == 1:
                if len(self._feature_names) != 1:
                    raise ValueError('Length of feature_names and X must be the same.')
            elif len(self._X.shape) == 2:
                if len(self._feature_names) != self._X.shape[1]:
                    raise ValueError('Length of feature_names and X must be the same.')
        if self._feature_names is None and self._X is not None:
            if len(self._X.shape) == 1:
                self._feature_names = np.array(['feature_0'])
            elif len(self._X.shape) == 2:
                self._feature_names = np.array([f'feature_{i}' for i in range(self._X.shape[1])])
        if self._label_names is not None and self._y is not None:
            if len(self._y.shape) == 1:
                if len(self._label_names) != 1:
                    raise ValueError('Length of label_names and y must be the same.')
            elif len(self._y.shape) == 2:
                if len(self._label_names) != self._y.shape[1]:
                    raise ValueError('Length of label_names and y must be the same.')
        if self._label_names is None and self._y is not None:
            if len(self._y.shape) == 1:
                self._label_names = np.array(['y'])
            elif len(self._y.shape) == 2:
                self._label_names = np.array([f'y_{i}' for i in range(self._y.shape[1])])

    def _reset(self, smiles: Union[np.ndarray, List[str]]) -> None:
        """
        Resets the dataset.
        Changes the smiles and updates the mols, ids, X and y.
        Parameters
        ----------
        smiles: Union[np.ndarray, List[str]]
            SMILES strings of the new molecules.
        """
        super().__init__()
        self._smiles = np.array(smiles)
        self._ids = np.array([str(uuid.uuid4().hex) for _ in range(len(smiles))])
        self._X = None
        self._y = None
        self._n_tasks = None
        self._mols = np.array([smiles_to_mol(s) for s in self._smiles])
        self.remove_elements([self._ids[i] for i, m in enumerate(self._mols) if m is None])
        self._feature_names = None
        self._label_names = None

    @property
    def smiles(self) -> np.ndarray:
        """
        Get the SMILES strings of the molecules in the dataset.
        Returns
        -------
        np.ndarray
            SMILES strings of the molecules in the dataset.
        """
        return self._smiles

    @smiles.setter
    def smiles(self, smiles: Union[np.ndarray, List[str]]) -> None:
        """
        Set the SMILES strings of the molecules in the dataset.
        Parameters
        ----------
        smiles: Union[np.ndarray, List[str]]
            SMILES strings of the molecules.
        """
        warnings.warn('The RDKit Mol objects of the dataset will be updated, IDs updated and X and y deleted.')
        self._reset(smiles)

    @property
    def mols(self) -> np.ndarray:
        """
        Get the RDKit Mol objects of the molecules in the dataset.
        Returns
        -------
        np.ndarray
            RDKit molecules of the molecules in the dataset.
        """
        return self._mols

    @property
    def feature_names(self) -> np.ndarray:
        """
        Get the feature labels of the molecules in the dataset.
        Returns
        -------
        np.ndarray
            Feature names of the molecules in the dataset.
        """
        return self._feature_names

    @feature_names.setter
    def feature_names(self, feature_names: Union[List, np.ndarray]) -> None:
        """
        Set the feature labels of the molecules in the dataset.
        Parameters
        ----------
        feature_names: Union[List, np.ndarray]
            Feature names of the molecules.
        """
        if self._X is None:
            raise ValueError('The features must be set before setting the feature names.')
        if len(self._X.shape) == 1:
            if len(feature_names) != 1:
                raise ValueError('The number of feature names must be equal to the number of features.')
        else:
            if len(feature_names) != len(self._X[0]):
                raise ValueError('The number of feature names must be equal to the number of features.')
        if len(feature_names) != len(set(feature_names)):
            raise ValueError('The feature names must be unique.')
        self._feature_names = np.array([str(fn) for fn in feature_names])

    @property
    def label_names(self) -> np.ndarray:
        """
        Get the label names of the molecules in the dataset.
        Returns
        -------
        np.ndarray
            Label names of the molecules in the dataset.
        """
        return self._label_names

    @label_names.setter
    def label_names(self, label_names: Union[List, np.ndarray]) -> None:
        """
        Set the label names of the molecules in the dataset.
        Parameters
        ----------
        label_names: Union[List, np.ndarray]
            Label names of the molecules.
        """
        if self._y is None:
            raise ValueError('The labels must be set before setting the label names.')
        if len(self._y.shape) == 1:
            if len(label_names) != 1:
                raise ValueError('The number of label names must be equal to the number of labels.')
        else:
            if len(label_names) != len(self._y[0]):
                raise ValueError('The number of label names must be equal to the number of labels.')
        if len(label_names) != len(set(label_names)):
            raise ValueError('The label names must be unique.')
        self._label_names = np.array([str(ln) for ln in label_names])

    @property
    def X(self) -> np.ndarray:
        """
        Get the features of the molecules in the dataset.
        Returns
        -------
        np.ndarray
            Features of the molecules in the dataset.
        """
        return self._X

    @property
    def y(self) -> np.ndarray:
        """
        Get the labels of the molecules in the dataset.
        Returns
        -------
        np.ndarray
            Labels of the molecules in the dataset.
        """
        return self._y

    @property
    def ids(self) -> np.ndarray:
        """
        Get the IDs of the molecules in the dataset.
        Returns
        -------
        np.ndarray
            IDs of the molecules in the dataset.
        """
        return self._ids

    @ids.setter
    def ids(self, ids: Union[List, np.ndarray]) -> None:
        """
        Set the IDs of the molecules in the dataset.
        Parameters
        ----------
        ids: Union[List, np.ndarray]
            IDs of the molecules.
        """
        if len(ids) != len(self._smiles):
            raise ValueError('The number of IDs must be equal to the number of molecules.')
        if len(ids) != len(np.unique(ids)):
            raise ValueError('The IDs must be unique.')
        self._ids = np.array([str(idx) for idx in ids])

    @property
    def n_tasks(self) -> int:
        """
        Get the number of tasks in the dataset.

        Returns
        -------
        n_tasks: int
            The number of tasks in the dataset.
        """
        return self._n_tasks

    def get_shape(self) -> Tuple[Tuple, Union[Tuple, None], Union[Tuple, None]]:
        """
        Get the shape of the dataset.
        Returns three tuples, giving the shape of the smiles, X and y arrays.
        Returns
        -------
        smiles_shape: Tuple
            The shape of the mols array.
        X_shape: Union[Tuple, None]
            The shape of the X array.
        y_shape: Union[Tuple, None]
            The shape of the y array.
        """
        smiles_shape = self._smiles.shape
        self.logger.info(f'Mols_shape: {smiles_shape}')
        x_shape = self._X.shape if self._X else None
        self.logger.info(f'Features_shape: {x_shape}')
        y_shape = self._y.shape if self._y else None
        self.logger.info(f'Labels_shape: {y_shape}')
        return smiles_shape, x_shape, y_shape

    def remove_duplicates(self) -> None:
        """
        Remove molecules with duplicated features from the dataset.
        """
        if self._X is not None:
            if np.isnan(np.stack(self._X)).any():
                warnings.warn('The dataset contains NaNs. Molecules with NaNs will be ignored.')
            unique, index = np.unique(self.X, return_index=True, axis=0)
            ids = self.ids[index]
            self.select(ids, axis=0)

    def remove_elements(self, ids: List[str]) -> None:
        """
        Remove elements with specific IDs from the dataset.
        Parameters
        ----------
        ids: List[str]
            IDs of the elements to remove.
        """
        if len(ids) != 0:
            all_indexes = self.ids
            indexes_to_keep = list(set(all_indexes) - set(ids))
            self.select(indexes_to_keep)

    def remove_elements_by_index(self, indexes: List[int]) -> None:
        """
        Remove elements with specific indexes from the dataset.
        Parameters
        ----------
        indexes: List[int]
            Indexes of the elements to remove.
        """
        if len(indexes) > 0:
            indexes = self._ids[indexes]
            self.remove_elements(indexes)

    def select_features_by_index(self, indexes: List[int]) -> None:
        """
        Select features with specific indexes from the dataset
        Parameters
        ----------
        indexes: List[int]
            The indexes of the features to select from the dataset.
        """
        if len(indexes) != 0:
            self.select(indexes, axis=1)

    def select_features_by_name(self, names: List[str]) -> None:
        """
        Select features with specific names from the dataset
        Parameters
        ----------
        names: List[str]
            The names of the features to select from the dataset.
        """
        if len(names) != 0:
            # Get the indexes of the features to select
            indexes = [i for i, name in enumerate(self._feature_names) if name in names]
            self.select(indexes, axis=1)

    def remove_nan(self, axis: int = 0) -> None:
        """
        Remove samples with at least one NaN in the features (when axis = 0)
        Or remove samples with all features with NaNs and the features with at least one NaN (axis = 1)
        Parameters
        ----------
        axis: int
            The axis to remove the NaNs from.
        """
        if self._X is None or len(self._X.shape) == 0:
            return
        if axis == 0:
            if len(self._X.shape) == 1:
                indexes = np.where(pd.isna(self._X))[0]
            else:
                indexes = np.where(pd.isna(self._X).any(axis=1))[0]
            # rows with at least one NaN
            self.remove_elements_by_index(indexes)
        elif axis == 1:
            if len(self._X.shape) == 1:
                indexes = np.where(np.isnan(self._X))[0]
                self.remove_elements_by_index(indexes)
            else:
                # rows with all NaNs
                indexes = np.where(np.isnan(self._X).all(axis=1))[0]
                self.remove_elements_by_index(indexes)
                # columns with at least one NaN
                columns = list(set(np.where(np.isnan(self._X).any(axis=0))[0]))
                self._X = np.delete(self._X, columns, axis=1)
                if len(self._X.shape) <= 2:  # feature names in datasets with more than two dimensions not supported
                    feature_names_to_delete = [self._feature_names[i] for i in columns]
                    self._feature_names = [name for name in self._feature_names if name not in feature_names_to_delete]
        else:
            raise ValueError('The axis must be 0 or 1.')

    def select_to_split(self, indexes: Union[np.ndarray, List[int]]) -> 'SmilesDataset':
        """
        Select elements with specific indexes to split the dataset
        Parameters
        ----------
        indexes: Union[np.ndarray, List[int]]
            The indexes of the elements to split the dataset.
        Returns
        -------
        SmilesDataset
            The dataset with the selected elements.
        """
        smiles = self._smiles[indexes]
        mols = self._mols[indexes]
        X = self._X[indexes] if self._X is not None else None
        y = self._y[indexes] if self._y is not None else None
        ids = self._ids[indexes]
        feature_names = self._feature_names
        return SmilesDataset(smiles, mols, ids, X, feature_names, y)

    def select(self, indexes: Union[List[str], List[int]], axis: int = 0) -> None:
        """
        Creates a new sub dataset of self from a selection of indexes.
        Parameters
        ----------
        indexes: Union[List[str], List[int]]
          List of ids/indexes to select.
          IDs of the compounds in case axis = 0, indexes of the columns in case axis = 1.
        axis: int
            Axis to select along. 0 selects along the first axis, 1 selects along the second axis.
        """
        if axis == 0:
            ids_to_delete = sorted(list(set(self._ids) - set(indexes)))
            raw_indexes = [i for i, mol_index in enumerate(self._ids) if mol_index in ids_to_delete]
            for idx in raw_indexes:
                self.logger.error(f"Molecule with smiles: {self._smiles[idx]} removed from dataset.")
            self._smiles = np.delete(self._smiles, raw_indexes, axis)
            self._mols = np.delete(self._mols, raw_indexes, axis)
            self._y = np.delete(self._y, raw_indexes, axis) if self._y is not None else self._y
            self._X = np.delete(self._X, raw_indexes, axis) if self._X is not None else self._X
            self._ids = np.delete(self._ids, raw_indexes, axis)

        elif axis == 1:
            if self._X is None or len(self._X.shape) == 0:
                raise ValueError('Dataset has no features.')
            if len(self._X.shape) == 1:
                pass
            else:
                indexes_to_delete = list(set(np.arange(self._X.shape[1])) - set(indexes))
                self._X = np.delete(self.X, indexes_to_delete, axis=1)
                if len(self._X.shape) <= 2:  # feature names in datasets with more than two dimensions not supported
                    feature_names_to_delete = [self._feature_names[i] for i in indexes_to_delete]
                    self._feature_names = [name for name in self._feature_names if name not in feature_names_to_delete]
        else:
            raise ValueError('The axis must be 0 or 1.')

    def merge(self, datasets: List[Dataset]) -> 'SmilesDataset':
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

        X = self._X
        y = self._y
        ids = self._ids
        mols = self._mols
        smiles = self._smiles
        feature_names = self._feature_names
        label_names = self._label_names

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
            smiles = np.append(smiles, ds.smiles, axis=0)
        return SmilesDataset(smiles, mols, ids, X, feature_names, y, label_names)

    def to_csv(self, path: str) -> None:
        """
        Save the dataset to a csv file.
        Parameters
        ----------
        path: str
            Path to save the csv file.
        """
        df = pd.DataFrame()
        df['ids'] = pd.Series(self._ids)
        df['smiles'] = pd.Series(self._smiles)
        if self._y is not None:
            label_names = self._label_names
            df_y = pd.DataFrame(self._y, columns=label_names)
            df = pd.concat([df, df_y], axis=1)
        if self._X is not None:
            columns_names = self._feature_names
            df_x = pd.DataFrame(self._X, columns=columns_names)
            df = pd.concat([df, df_x], axis=1)

        df.to_csv(path, index=False)

    def load_features(self, path: str, **kwargs) -> None:
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
        self._X = df.to_numpy()

    def save_features(self, path: str = 'features.csv') -> None:
        """
        Save the features to a csv file.
        Parameters
        ----------
        path: str
            Path to save the csv file.
        """
        if self.X is not None:
            columns_names = self._feature_names
            df = pd.DataFrame(self._X, columns=columns_names)
            df.to_csv(path, index=False)
        else:
            raise ValueError('Features array is empty!')
