"""
Classes for processing input data into a format suitable for machine learning.
"""

from typing import Optional, Union, List

from deepmol.datasets import NumpyDataset
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
                 mols_field: str,
                 id_field: str = None,
                 labels_fields: Union[List[str], str] = None,
                 features_fields: Union[List[str], str] = None,
                 features2keep: List[Union[str, int]] = None,
                 shard_size: int = None) -> None:
        """
        Initialize the CSVLoader.

        Parameters
        ----------
        dataset_path: str
            path to the dataset file
        mols_field: str
            field containing the molecules'
        id_field: str
            field containing the ids
        labels_fields: Union[List[str], str]
            field containing the labels
        features_fields: Union[List[str], str]
            field containing the features
        features2keep: List[Union[str, int]]
            features to keep
        shard_size: int
            size of the shard to load
        """
        self.dataset_path = dataset_path
        self.mols_field = mols_field
        self.id_field = id_field
        self.labels_fields = labels_fields
        self.features_fields = features_fields

        self.features2keep = features2keep
        self.shard_size = shard_size

        fields2keep = [mols_field]

        if id_field is not None:
            fields2keep.append(id_field)

        if labels_fields is not None:
            self.n_tasks = len(labels_fields)
            if isinstance(labels_fields, list):
                [fields2keep.append(x) for x in labels_fields]
            else:
                fields2keep.append(labels_fields)
        else:
            self.n_tasks = 0

        if features_fields is not None:
            if isinstance(features_fields, list):
                [fields2keep.append(x) for x in features_fields]
            else:
                fields2keep.append(features_fields)

        self.fields2keep = fields2keep

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

    def create_dataset(self, **kwargs) -> NumpyDataset:
        """
        Creates a dataset from the CSV file.

        Parameters
        ----------
        kwargs:
            Keyword arguments to pass to pandas.read_csv.

        Returns
        -------
        NumpyDataset
            Dataset with the data.
        """
        dataset = self._get_dataset(self.dataset_path, fields=self.fields2keep, chunk_size=self.shard_size, **kwargs)

        mols = dataset[self.mols_field].to_numpy()

        if self.features_fields is not None:
            X = dataset[self.features_fields].to_numpy()
        else:
            X = None

        if self.labels_fields is not None:
            y = dataset[self.labels_fields].to_numpy()
        else:
            y = None

        if self.id_field is not None:
            ids = dataset[self.id_field].to_numpy()
        else:
            ids = np.array([i for i in range(dataset.shape[0])])

        return NumpyDataset(mols=mols,
                            X=X,
                            y=y,
                            ids=ids,
                            features2keep=self.features2keep,
                            n_tasks=self.n_tasks)


class SDFLoader(object):
    """
    A Loader to directly read data from a SDF.
    """

    def __init__(self,
                 dataset_path: str,
                 id_field: str = None,
                 labels_fields: Union[List[str], str] = None,
                 features_fields: Union[List[str], str] = None,
                 features2keep: List[Union[str, int]] = None,
                 shard_size: Optional[int] = None) -> None:
        """
        Initialize the SDFLoader.

        Parameters
        ----------
        dataset_path: str
            path to the dataset file
        id_field: str
            field containing the ids
        labels_fields: Union[List[str], str]
            field containing the labels
        features_fields: Union[List[str], str]
            field containing the features
        features2keep: List[Union[str, int]]
            features to keep
        shard_size: int
            size of the shard to load
        """
        self.dataset_path = dataset_path
        self.id_field = id_field
        self.labels_fields = labels_fields
        self.features_fields = features_fields
        self.features2keep = features2keep
        self.shard_size = shard_size

        fields2keep = []

        if id_field is not None:
            fields2keep.append(id_field)

        if labels_fields is not None:
            self.n_tasks = len(labels_fields)
            if isinstance(labels_fields, list):
                [fields2keep.append(x) for x in labels_fields]
            else:
                fields2keep.append(labels_fields)
        else:
            self.n_tasks = 0

        if features_fields is not None:
            if isinstance(features_fields, list):
                [fields2keep.append(x) for x in features_fields]
            else:
                fields2keep.append(features_fields)

        self.fields2keep = fields2keep

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

    def create_dataset(self) -> NumpyDataset:
        """
        Creates a dataset from the SDF file.

        Returns
        -------
        NumpyDataset
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
            if self.features_fields is not None:
                if isinstance(self.features_fields, list):
                    for feature in self.features_fields:
                        mol_feature.append(mol.GetProp(feature))
                else:
                    mol_feature.append(mol.GetProp(self.features_fields))

                X.append(mol_feature)

            if self.labels_fields is not None:

                if isinstance(self.labels_fields, list):
                    if len(self.labels_fields) == 1:
                        y.append(float(mol.GetProp(self.labels_fields[0])))
                    else:
                        mol_ys = []
                        for label in self.labels_fields:
                            mol_ys.append(float(mol.GetProp(label)))

                        y.append(mol_ys)
                else:
                    y.append(float(mol.GetProp(self.labels_fields)))
            else:
                mol_y = None
                y.append(mol_y)

            if self.id_field is not None:
                mol_id = mol.GetProp(self.id_field)
                ids.append(mol_id)
            else:
                mol_id = None
                ids.append(mol_id)

        X = np.array(X)
        y = np.array(y)
        ids = np.array(ids)
        mols = np.array(mols)

        return NumpyDataset(mols=mols,
                            X=X,
                            y=y,
                            ids=ids,
                            features2keep=self.features2keep,
                            n_tasks=self.n_tasks)
