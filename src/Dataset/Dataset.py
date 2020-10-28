import numpy as np
import pandas as pd
from typing import Tuple, Iterator, Optional, Sequence

Batch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def load_csv_file(input_file, keep_fields, chunk_size=None):
    """Load data as pandas dataframe from CSV files.
    Parameters
    ----------
    input_files: str
        data path
    chunk_size: int, default None
        The chunk size to yield at one time.
    Returns
    -------
    Iterator[pd.DataFrame]
        Generator which yields the dataframe which is the same chunk size.
    """

    chunk_num = 1
    if chunk_size is None:
        if keep_fields is None:
            return pd.read_csv(input_file)
        else : return pd.read_csv(input_file)[keep_fields]
    else:
        #TODO: fix code to return datasset divided in chunks
        for df in pd.read_csv(input_file, chunksize=chunk_size):
            print("Loading shard %d of size %s." % (chunk_num, str(chunk_size)))
            df = df.replace(np.nan, str(""), regex=True)
            chunk_num += 1
            if keep_fields is None:
                return df
            else : return df[keep_fields]

class Dataset(object):
    """Abstract base class for datasets
    """

    def __init__(self) -> None:
        raise NotImplementedError()

    def __len__(self) -> int:
        """Get the number of elements in the dataset."""
        raise NotImplementedError()

    def get_shape(self) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """Get the shape of all the elements of the dataset.
        Returns three tuples, shape of X (number of examples), y (number of tasks) and ids arrays.
        """
        raise NotImplementedError()

    def X(self) -> np.ndarray:
        """Get the X (number of examples) vector for this dataset as a single numpy array."""
        raise NotImplementedError()

    def y(self) -> np.ndarray:
        """Get the y (number of tasks) vector for this dataset as a single numpy array."""
        raise NotImplementedError()

    def ids(self) -> np.ndarray:
        """Get the ids vector for this dataset as a single numpy array."""
        raise NotImplementedError()

    def features(self) -> np.ndarray:
        """Get the features array for this dataset as a single numpy array."""
        raise NotImplementedError()

    def iterbatches(self, 
                    batch_size: Optional[int] = None,
                    epochs: int = 1,
                    deterministic: bool = False,
                    pad_batches: bool = False) -> Iterator[Batch]:
        """Get an object that iterates over minibatches from the dataset.
        Each minibatch is returned as a tuple of three numpy arrays:
        (X, y, ids).

        Parameters
        ----------
        batch_size: int, optional (default None)
        Number of elements in each batch.
        epochs: int, optional (default 1)
        Number of epochs to walk over dataset.
        deterministic: bool, optional (default False)
        If True, follow deterministic order.
        pad_batches: bool, optional (default False)
        If True, pad each batch to `batch_size`.
        Returns
        -------
        Iterator[Batch]
        Generator which yields tuples of four numpy arrays `(X, y, ids)`.
        """
        raise NotImplementedError("Each dataset model must implement its own iterbatches method.")

    def itersamples(self) -> Iterator[Batch]:
        """Get an object that iterates over the samples in the dataset."""
        raise NotImplementedError("Each dataset model must implement its own itersamples method.")


class NumpyDataset(Dataset):
    """A Dataset defined by in-memory numpy arrays.
      This subclass of `Dataset` stores arrays `X,y,features,ids` in memory as
      numpy arrays. This makes it very easy to construct `NumpyDataset`
      objects.
      """

    def __init__(self,
                 X: np.ndarray,
                 y: Optional[np.ndarray] = None,
                 features: Optional[np.ndarray] = None,
                 ids: Optional[np.ndarray] = None,
                 n_tasks: int = 1) -> None:
        """Initialize this object.
        Parameters
        ----------
        X: np.ndarray
          Input features. A numpy array of shape `(n_samples,...)`.
        y: np.ndarray, optional (default None)
          Labels. A numpy array of shape `(n_samples, ...)`. Note that each label can
          have an arbitrary shape.
        w: np.ndarray, optional (default None)
          Weights. Should either be 1D array of shape `(n_samples,)` or if
          there's more than one task, of shape `(n_samples, n_tasks)`.
        ids: np.ndarray, optional (default None)
          Identifiers. A numpy array of shape `(n_samples,)`
        n_tasks: int, default 1
          Number of learning tasks.
        """
        n_samples = len(X)
        if n_samples > 0:
            if y is None:
                # Set labels to be zero, with zero weights
                y = np.zeros((n_samples, n_tasks), np.float32)
                w = np.zeros((n_samples, 1), np.float32)
        if ids is None:
            ids = np.arange(n_samples)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if features is None:
            if len(y.shape) == 1:
                features = np.ones(y.shape[0], np.float32)
            else:
                features = np.ones((y.shape[0], 1), np.float32)
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        self.X = X
        self.y = y
        self.features = features
        self.ids = np.array(ids, dtype=object)

    def __len__(self) -> int:
        """Get the number of elements in the dataset."""
        return len(self.y)

    def get_shape(self) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """Get the shape of the dataset.
        Returns four tuples, giving the shape of the X, y, w, and ids arrays.
        """
        return self.X.shape, self.y.shape, self.w.shape, self.ids.shape

    def get_task_names(self) -> np.ndarray:
        """Get the names of the tasks associated with this dataset."""
        if len(self.y.shape) < 2:
            return np.array([0])
        return np.arange(self.y.shape[1])


    def X(self) -> np.ndarray:
        """Get the X vector for this dataset as a single numpy array."""
        return self.X


    def y(self) -> np.ndarray:
        """Get the y vector for this dataset as a single numpy array."""
        return self.y


    def ids(self) -> np.ndarray:
        """Get the ids vector for this dataset as a single numpy array."""
        return self.ids


    def features(self) -> np.ndarray:
        """Get the features array for this dataset as a single numpy array."""
        return self.features



class CSVLoader(Dataset):
    '''
    ...
    '''

    def __init__(self, dataset_path, input_field, output_fields=None,
                 id_field=None, user_features=None, keep_all_fields=False,
                 chunk_size=None):
        """Initialize this object.
        Parameters
        ----------
        dataset_path: string
            path to the dataset file
        input_field: string
            label of the input feature (smiles, inchi, etc)
        output_features: list of strings (optional, default: None)
            labels of the output features
        id_filed: string (optional, default: None)
            label of the input additional identifiers
        user_features: list of strings (optional, default: None)
            label of the user provided features
        keep_all_fields: Boolean (optional, default: False)
        """

        if not isinstance(dataset_path, str):
            raise ValueError("Dataset path must be a string")

        if not isinstance(input_field, str):
            raise ValueError("Input identifier must be a string")

        if not isinstance(output_fields, list) and output_fields is not None:
            raise ValueError("Output fields must be a list or None.")

        if not isinstance(id_field, str) and id_field is not None:
            raise ValueError("Field id must be a string or None.")

        #TODO: provide user features in a better way
        if not isinstance(user_features, list) and user_features is not None:
            raise ValueError("User features must be a list of string containing "
                             "the features fields or None.")

        self.dataset_path = dataset_path
        self.tasks = output_fields
        self.input_field = input_field
        self.chunk_size = chunk_size

        if id_field is None:
            self.id_field = input_field
        else:
            self.id_field = id_field

        self.user_features = user_features

        if keep_all_fields:
            self.dataset = self._get_dataset(dataset_path, keep_fields=None, chunk_size = self.chunk_size)
        else:
            columns = [id_field, input_field]
            for field in output_fields:
                columns.append(field)
            if user_features is None:
                self.dataset = self._get_dataset(dataset_path, keep_fields=columns, chunk_size = self.chunk_size)
            else:
                columns = columns + user_features
                self.dataset = self._get_dataset(dataset_path, keep_fields=columns, chunk_size = self.chunk_size)

        self.X = self.dataset[input_field]

        self.y = self.dataset[output_fields]

        self.ids = self.dataset[id_field]


        if len(self.X) > 0:
            if not isinstance(self.X, np.ndarray):
                self.X = np.array(self.X)
            if not isinstance(self.ids, np.ndarray):
                self.ids = np.array(self.ids, dtype=object)

            if self.y is None:
                self.y = np.zeros((len(self.X), self.n_tasks), np.float32)
            else :
                if self.y.shape[1] < 2:
                    if not isinstance(self.y[output_fields[0]], np.ndarray):
                        self.y = np.array(self.y[output_fields[0]])
                else:
                    output_arrays = []
                    for field in output_fields:
                        if not isinstance(self.y[field], np.ndarray):
                            output_arrays.append(np.array(self.y[field]))
                        else:
                            output_arrays.append(self.y[field])
                    self.y = output_arrays


        else :
            raise ValueError("Invalid or Empty Dataset!")

    def __len__(self):
        """Get the number of elements in the dataset."""
        return len(self.y)

    def get_shape(self):
        """Get the shape of the dataset.
        Returns four tuples, giving the shape of the X, y, features, and ids arrays.
        """
        return self.X.shape, self.y.shape, self.features.shape, self.ids.shape

    def get_task_names(self):
        """Get the names of the tasks associated with this dataset."""
        if len(self.y.shape) < 2:
            return np.array([0])
        return np.arange(self.y.shape[1])


    def X(self):
        """Get the X vector for this dataset as a single numpy array."""
        return self.X

    def y(self):
        """Get the y vector for this dataset as a single numpy array."""
        return self.y

    def ids(self):
        """Get the ids vector for this dataset as a single numpy array."""
        return self.ids

    def features(self):
        """Get the features vector for this dataset as a single numpy array."""
        return self.features

    def _get_dataset(self, dataset_path, keep_fields=None, chunk_size=None):
        """Defines a generator which returns data for each shard
        Parameters
        ----------
        input_files: List[str]
          List of filenames to process
        shard_size: int, optional
          The size of a shard of data to process at a time.
        Returns
        -------
        Iterator[pd.DataFrame]
          Iterator over shards
        """
        return load_csv_file(dataset_path, keep_fields, chunk_size)

    def select(self, indices: Sequence[int]) -> Dataset:
        """Creates a new dataset from a selection of indices from self.
        Parameters
        ----------
        indices: List[int]
          List of indices to select.
        select_dir: str, optional (default None)
          Used to provide same API as `DiskDataset`. Ignored since
          `NumpyDataset` is purely in-memory.
        Returns
        -------
        Dataset
          A selected Dataset object
        """
        X = self.X[indices]
        y = self.y[indices]
        features = self.features[indices]
        ids = self.ids[indices]
        return NumpyDataset(X, y, features, ids)
