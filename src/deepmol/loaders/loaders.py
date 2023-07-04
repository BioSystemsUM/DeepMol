"""
Classes for processing input data into a format suitable for machine learning.
"""

from typing import Optional, List, Union

from deepmol.datasets import SmilesDataset
import numpy as np
import pandas as pd

from deepmol.loaders._utils import load_csv_file, load_sdf_file


class CSVLoader(object):
    """
    A Loader to directly read data from a CSV.
    Assumes a coma separated with header file!
    """

    def __init__(self,
                 dataset_path: str,
                 smiles_field: str,
                 id_field: str = None,
                 labels_fields: List[str] = None,
                 features_fields: List[str] = None,
                 shard_size: int = None,
                 mode: Union[str, List[str]] = 'auto') -> None:
        """
        Initialize the CSVLoader.

        Parameters
        ----------
        dataset_path: str
            path to the dataset file
        smiles_field: str
            field containing the molecules'
        id_field: str
            field containing the ids
        labels_fields: List[str]
            field containing the labels
        features_fields: List[str]
            field containing the features
        shard_size: int
            size of the shard to load
        mode: Union[str, List[str]]
            The mode of the dataset.
            If 'auto', the mode is inferred from the labels. If 'classification', the dataset is treated as a
            classification dataset. If 'regression', the dataset is treated as a regression dataset. If a list of
            modes is passed, the dataset is treated as a multi-task dataset.
        """
        self.dataset_path = dataset_path
        self.mols_field = smiles_field
        self.id_field = id_field
        self.labels_fields = labels_fields
        self.features_fields = features_fields
        self.shard_size = shard_size
        fields2keep = [smiles_field]

        if id_field is not None:
            fields2keep.append(id_field)

        if labels_fields is not None:
            fields2keep.extend(labels_fields)

        if features_fields is not None:
            fields2keep.extend(features_fields)

        self.fields2keep = fields2keep
        self.mode = mode

    @staticmethod
    def _get_dataset(dataset_path: str,
                     fields: List[str] = None,
                     chunk_size: int = None,
                     **kwargs) -> pd.DataFrame:
        """
        Loads data with size chunk_size.

        Parameters
        ----------
        dataset_path: str
            path to the dataset file
        fields: List[str]
            fields to keep
        chunk_size: int
            size of the shard to load
        kwargs:
            Keyword arguments to pass to pandas.read_csv.

        Returns
        -------
        pd.DataFrame
            Dataframe with chunk size.
        """
        return load_csv_file(dataset_path, fields, chunk_size, **kwargs)

    def create_dataset(self, **kwargs) -> SmilesDataset:
        """
        Creates a dataset from the CSV file.

        Parameters
        ----------
        kwargs:
            Keyword arguments to pass to pandas.read_csv.

        Returns
        -------
        SmilesDataset
            Dataset with the data.
        """
        dataset = self._get_dataset(self.dataset_path, fields=self.fields2keep, chunk_size=self.shard_size, **kwargs)

        mols = dataset[self.mols_field].to_numpy()

        if self.features_fields is not None:
            if len(self.features_fields) == 1:
                X = dataset[self.features_fields[0]].to_numpy()
            else:
                X = dataset[self.features_fields].to_numpy()
        else:
            X = None

        if self.labels_fields is not None:
            if len(self.labels_fields) == 1:
                y = dataset[self.labels_fields[0]].to_numpy()
            else:
                y = dataset[self.labels_fields].to_numpy()
        else:
            y = None

        if self.id_field is not None:
            ids = dataset[self.id_field].to_numpy()
        else:
            ids = None

        return SmilesDataset(smiles=mols,
                             X=X,
                             y=y,
                             ids=ids,
                             feature_names=self.features_fields,
                             label_names=self.labels_fields,
                             mode=self.mode)


class SDFLoader(object):
    """
    A Loader to directly read data from a SDF.
    """

    def __init__(self,
                 dataset_path: str,
                 id_field: str = None,
                 labels_fields: List[str] = None,
                 features_fields: List[str] = None,
                 shard_size: Optional[int] = None,
                 mode: Union[str, List[str]] = 'auto') -> None:
        """
        Initialize the SDFLoader.

        Parameters
        ----------
        dataset_path: str
            path to the dataset file
        id_field: str
            field containing the ids
        labels_fields: List[str]
            field containing the labels
        features_fields: List[str]
            field containing the features
        shard_size: int
            size of the shard to load
        mode: Union[str, List[str]]
            The mode of the dataset.
            If 'auto', the mode is inferred from the labels. If 'classification', the dataset is treated as a
            classification dataset. If 'regression', the dataset is treated as a regression dataset. If a list of
            modes is passed, the dataset is treated as a multi-task dataset.
        """
        self.dataset_path = dataset_path
        self.id_field = id_field
        self.labels_fields = labels_fields
        self.features_fields = features_fields
        self.shard_size = shard_size

        fields2keep = []

        if id_field is not None:
            fields2keep.append(id_field)

        if labels_fields is not None:
            fields2keep.extend(labels_fields)

        if features_fields is not None:
            fields2keep.extend(features_fields)

        self.fields2keep = fields2keep
        self.mode = mode

    @staticmethod
    def _get_dataset(dataset_path: str, chunk_size: int = None) -> np.ndarray:
        """
        Loads data from path.

        Parameters
        ----------
        dataset_path: str
            Filename to process
        chunk_size: int
            size of the shard to load
        Returns
        -------
        pd.DataFrame
            Dataframe
        """
        return load_sdf_file(dataset_path, chunk_size)

    def create_dataset(self) -> SmilesDataset:
        """
        Creates a dataset from the SDF file.

        Returns
        -------
        SmilesDataset
            Dataset with the data.
        """
        molecules = self._get_dataset(self.dataset_path, self.shard_size)
        X = []
        y = []
        ids = []
        mols = []
        for mol in molecules:
            mols.append(mol)
            mol_feature = []
            mol_ys = []
            if self.features_fields is not None:
                for feature in self.features_fields:
                    mol_feature.append(mol.GetProp(feature))
                if len(mol_feature) == 1:
                    mol_feature = mol_feature[0]
                X.append(mol_feature)

            if self.labels_fields is not None:
                for label in self.labels_fields:
                    mol_ys.append(float(mol.GetProp(label)))
                if len(mol_ys) == 1:
                    mol_ys = mol_ys[0]
                y.append(mol_ys)
            else:
                mol_y = None
                y.append(mol_y)

            if self.id_field is not None:
                mol_id = mol.GetProp(self.id_field)
                ids.append(mol_id)
            else:
                mol_id = None
                ids.append(mol_id)

        X = np.array(X) if X is not None and len(X) != 0 else None
        y = None if len(set(np.array(y).flatten())) == 1 and np.array(y).flatten()[0] is None else np.array(y)
        ids = np.array(ids) if len(set(ids)) == len(ids) else None
        mols = np.array(mols)
        feature_names = self.features_fields
        return SmilesDataset.from_mols(mols=mols,
                                       X=X,
                                       y=y,
                                       ids=ids,
                                       feature_names=feature_names,
                                       label_names=self.labels_fields,
                                       mode=self.mode)
