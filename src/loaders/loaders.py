"""
Classes for processing input data into a format suitable for machine learning.
"""

from typing import Optional, Union, List

from rdkit.Chem import SDMolSupplier

from datasets.datasets import NumpyDataset
import numpy as np
import pandas as pd


def load_csv_file(input_file, fields, sep=',', header=0, chunk_size=None):
    """Load data as pandas dataframe from CSV files.
    Parameters
    ----------
    input_file: str
        data path
    fields: np.ndarray
        fields to keep
    sep: str
        separator
    header: int
        the row where the header is
    chunk_size: int, default None
        The chunk size to yield at one time.
    Returns
    -------
    pd.DataFrame
        Dataframe with size chunk size.
    """

    if chunk_size is None:
        return pd.read_csv(input_file, sep=sep, header=header)[fields]
    else:
        df = pd.read_csv(input_file)
        print("Loading shard of size %s." % (str(chunk_size)))
        df = df.replace(np.nan, str(""), regex=True)
        return df[fields].sample(chunk_size)


def load_sdf_file(input_file):
    supplier = SDMolSupplier(input_file)
    mols, attempts = [], 0

    while not mols and attempts < 10:
        mols = list(supplier)
        attempts += 1
    print(f"Loaded {len(mols)} molecules after {attempts} attempts.")

    return mols


"""
class BaseDataLoader(object):
    '''bla bla
    '''

    def __init__(self,
                 data_path: str,
                 mols_field: str,
                 id_field: Optional[str] = None,
                 labels_fields: Optional[Union[List[str], str]] = None,
                 features_fields: Optional[Union[List[str], str]] = None,
                 features2keep: Optional[np.ndarray] = None,
                 shard_size: Optional[int] = None,
                 in_memory: Optional[bool] = True,
                 log_every_n: int = 1000):
        ''' Construct a BaseDataLoader object.

        #...

        '''

        if self.__class__ is BaseDataLoader:
            raise ValueError(
                "BaseDataLoader should never be instantiated directly. Use a subclass instead."
                )

        self.data_path = data_path
        self.mols_field = mols_field
        self.id_field = id_field
        self.labels_fields = labels_fields
        self.features_fields = features_fields
        self.features2keep = features2keep
        self.shard_size = shard_size

        if in_memory:
            self.dataset_type = 'numpy'
        else :
            #TODDO: implement a dataset class to store datasets in disk instead of memory
            self.dataset_type = 'disk'
            raise NotImplementedError
        self.log_every_n = log_every_n


    def create_dataset(self) -> NumpyDataset:
        '''bla bla
        '''
        raise NotImplementedError    

"""


class CSVLoader(object):
    """A Loader to directly read data from a CSV.
    Assumes a coma separated with header file!
    """

    def __init__(self,
                 dataset_path: str,
                 mols_field: str,
                 id_field: Optional[str] = None,
                 labels_fields: Optional[Union[List[str], str]] = None,
                 features_fields: Optional[Union[List[str], str]] = None,
                 features2keep: Optional[np.ndarray] = None,
                 shard_size: Optional[int] = None):
        """Initialize this object.
        Parameters
        ----------
        dataset_path: string
            path to the dataset file
        ...
        """

        if not isinstance(dataset_path, str):
            raise ValueError("Dataset path must be a string")

        if not isinstance(mols_field, str):
            raise ValueError("Input identifier must be a string")

        if (not isinstance(labels_fields, list) and not isinstance(labels_fields, str)) and labels_fields is not None:
            raise ValueError("Labels fields must be a str, list or None.")

        if not isinstance(id_field, str) and id_field is not None:
            raise ValueError("Field id must be a string or None.")

        if (not isinstance(features_fields, list) and not isinstance(features_fields,
                                                                     str)) and features_fields is not None:
            raise ValueError("User features must be a list of string containing "
                             "the features fields or None.")

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
    def _get_dataset(dataset_path, fields=None, sep=',', header=0, chunk_size=None):
        """Loads data with size chunk_size.
        Parameters
        ----------
        dataset_path: str
            Filename to process
        fields
        chunk_size: int, optional
            The size of a shard of data to process at a time.
        sep: str
            separator
        header: int
            the row where the header is
        fields: np.ndarray
            fields to keep
        Returns
        -------
        pd.DataFrame
            Dataframe
        """
        return load_csv_file(dataset_path, fields, sep, header, chunk_size)

    def create_dataset(self, sep=',', header=0, in_memory=True):
        if in_memory:
            dataset = self._get_dataset(self.dataset_path, fields=self.fields2keep, sep=sep, header=header,
                                        chunk_size=self.shard_size)

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

        else:
            raise NotImplementedError


class SDFLoader(object):
    """A Loader to directly read data from a SDF.
        """

    def __init__(self,
                 dataset_path: str,
                 id_field: Optional[str] = None,
                 labels_fields: Optional[Union[List[str], str]] = None,
                 features_fields: Optional[Union[List[str], str]] = None,
                 features2keep: Optional[np.ndarray] = None,
                 shard_size: Optional[int] = None):
        """Initialize this object.
        Parameters
        ----------
        dataset_path: string
            path to the dataset file
        ...
        """

        if not isinstance(dataset_path, str):
            raise ValueError("Dataset path must be a string")

        if (not isinstance(labels_fields, list) and not isinstance(labels_fields, str)) and labels_fields is not None:
            raise ValueError("Labels fields must be a str, list or None.")

        if not isinstance(id_field, str) and id_field is not None:
            raise ValueError("Field id must be a string or None.")

        if (not isinstance(features_fields, list) and not isinstance(features_fields,
                                                                     str)) and features_fields is not None:
            raise ValueError("User features must be a list of string containing "
                             "the features fields or None.")

        self._mols_handler = None

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

    @property
    def mols_handler(self):
        return self._mols_handler

    @mols_handler.setter
    def mols_handler(self, value):
        self._mols_handler = value

    @staticmethod
    def _get_dataset(dataset_path):
        """Loads data from path.
        Parameters
        ----------
        dataset_path: str
            Filename to process
        Returns
        -------
        pd.DataFrame
            Dataframe
        """
        return load_sdf_file(dataset_path)

    def create_dataset(self, in_memory=True):
        if in_memory:
            self.mols_handler = self._get_dataset(self.dataset_path)
            X = []
            y = []
            ids = []
            mols = []
            for mol in self.mols_handler:

                mols.append(mol)
                mol_feature = []
                if self.features_fields is not None:

                    for feature in self.features_fields:
                        mol_feature.append(mol.GetProp(feature))

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

        else:
            raise NotImplementedError


"""
class DatabaseLoader(BaseDataLoader):
    '''bla bla
    '''

    def __init__(self):
        super().__init__()
        

class JSONLoader(BaseDataLoader):
    '''bla bla
    '''

    def __init__(self):
        super().__init__()


class SDFLoader(BaseDataLoader):
    '''bla bla
    '''

    def __init__(self):
        super().__init__()

class FASTALoader(BaseDataLoader):
    '''bla bla
    '''

    def __init__(self):
        super().__init__()


class ImageLoader(BaseDataLoader):
    '''bla bla
    '''

    def __init__(self):
        super().__init__()

"""
