import numpy as np
import pandas as pd
from typing import Tuple, Iterator, Optional, Sequence, List, Iterable
import os
import tempfile
import time
import json
from utils.utils import save_to_disk

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

    def removeNAs(self):
        """Remove samples with NAs."""
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

    def select(self, indices: Sequence[int]) -> 'Dataset':
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
        features2keep = self.features2keep
        return NumpyDataset(X, y, features, ids, features2keep)


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
                 features2keep: Optional[List] = None,
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
        self.features2keep = features2keep

    def __len__(self) -> int:
        """Get the number of elements in the dataset."""
        return len(self.y)

    def get_shape(self) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """Get the shape of the dataset.
        Returns four tuples, giving the shape of the X, y, w, and ids arrays.
        """
        return self.X.shape, self.y.shape, self.features.shape, self.ids.shape

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

    def write_data_to_disk(data_dir: str,
                           basename: str,
                           tasks: np.ndarray,
                           X: Optional[np.ndarray] = None,
                           y: Optional[np.ndarray] = None,
                           features: Optional[np.ndarray] = None,
                           ids: Optional[np.ndarray] = None) -> List[Optional[str]]:
        """Static helper method to write data to disk.
        This helper method is used to write a shard of data to disk.
        Parameters
        ----------
        data_dir: str
          Data directory to write shard to.
        basename: str
          Basename for the shard in question.
        tasks: np.ndarray
          The names of the tasks in question.
        X: np.ndarray, optional (default None)
          The samples array.
        y: np.ndarray, optional (default None)
          The labels array.
        features: np.ndarray, optional (default None)
          The features array.
        ids: np.ndarray, optional (default None)
          The identifiers array.
        Returns
        -------
        List[Optional[str]]
          List with values `[out_ids, out_X, out_y, out_w, out_ids_shape,
          out_X_shape, out_y_shape, out_w_shape]` with filenames of locations to
          disk which these respective arrays were written.
        """
        if X is not None:
            out_X: Optional[str] = "%s-X.npy" % basename
            save_to_disk(X, os.path.join(data_dir, out_X))
            out_X_shape = X.shape
        else:
            out_X = None
            out_X_shape = None

        if y is not None:
            out_y: Optional[str] = "%s-y.npy" % basename
            save_to_disk(y, os.path.join(data_dir, out_y))
            out_y_shape = y.shape
        else:
            out_y = None
            out_y_shape = None

        if features is not None:
            out_features: Optional[str] = "%s-w.npy" % basename
            save_to_disk(features, os.path.join(data_dir, out_features))
            out_features_shape = features.shape
        else:
            out_features = None
            out_features_shape = None

        if ids is not None:
            out_ids: Optional[str] = "%s-ids.npy" % basename
            save_to_disk(ids, os.path.join(data_dir, out_ids))
            out_ids_shape = ids.shape
        else:
            out_ids = None
            out_ids_shape = None

        # note that this corresponds to the _construct_metadata column order
        return [
            out_ids, out_X, out_y, out_features, out_ids_shape, out_X_shape, out_y_shape,
            out_features_shape
        ]



    def create_dataset(shard_generator: Iterable[Batch],
                       data_dir: Optional[str] = None,
                       tasks: Optional[Sequence] = []) -> "NumpyDataset":
        """Creates a new DiskDataset
        Parameters
        ----------
        shard_generator: Iterable[Batch]
          An iterable (either a list or generator) that provides tuples of data
          (X, y, features, ids). Each tuple will be written to a separate shard on disk.
        data_dir: str, optional (default None)
          Filename for data directory. Creates a temp directory if none specified.
        tasks: Sequence, optional (default [])
          List of tasks for this dataset.
        Returns
        -------
        DiskDataset
          A new `DiskDataset` constructed from the given data
        """
        if data_dir is None:
            data_dir = tempfile.mkdtemp()
        elif not os.path.exists(data_dir):
            os.makedirs(data_dir)

        metadata_rows = []
        time1 = time.time()
        for shard_num, (X, y, features, ids) in enumerate(shard_generator):
            basename = "shard-%d" % shard_num
            metadata_rows.append(NumpyDataset.write_data_to_disk(data_dir, basename, tasks, X, y, features, ids))

        metadata_df = NumpyDataset._construct_metadata(metadata_rows)
        NumpyDataset._save_metadata(metadata_df, data_dir, tasks)
        time2 = time.time()
        print("TIMING: dataset construction took %0.3f s" % (time2 - time1))
        return NumpyDataset(data_dir)

    def _construct_metadata(metadata_entries: List) -> pd.DataFrame:
        """Construct a dataframe containing metadata.
        Parameters
        ----------
        metadata_entries: List
          `metadata_entries` should have elements returned by write_data_to_disk
          above.
        Returns
        -------
        pd.DataFrame
          A Pandas Dataframe object contains metadata.
        """
        columns = ('ids', 'X', 'y', 'features', 'ids_shape', 'X_shape', 'y_shape',
                   'features_shape')
        metadata_df = pd.DataFrame(metadata_entries, columns=columns)
        return metadata_df

    def _save_metadata(metadata_df: pd.DataFrame, data_dir: str,
                       tasks: Optional[Sequence]) -> None:
        """Saves the metadata for a DiskDataset
        Parameters
        ----------
        metadata_df: pd.DataFrame
          The dataframe which will be written to disk.
        data_dir: str
          Directory to store metadata.
        tasks: Sequence, optional
          Tasks of DiskDataset. If `None`, an empty list of tasks is written to
          disk.
        """
        if tasks is None:
            tasks = []
        elif isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        metadata_filename = os.path.join(data_dir, "metadata.csv.gzip")
        tasks_filename = os.path.join(data_dir, "tasks.json")
        with open(tasks_filename, 'w') as fout:
            json.dump(tasks, fout)
        metadata_df.to_csv(metadata_filename, index=False, compression='gzip')

    def merge(datasets: Iterable["Dataset"],
              merge_dir: Optional[str] = None) -> "NumpyDataset":
        """Merges provided datasets into a merged dataset.
        Parameters
        ----------
        datasets: Iterable[Dataset]
          List of datasets to merge.
        merge_dir: str, optional (default None)
          The new directory path to store the merged DiskDataset.
        Returns
        -------
        DiskDataset
          A merged DiskDataset.
        """
        if merge_dir is not None:
            if not os.path.exists(merge_dir):
                os.makedirs(merge_dir)
        else:
            merge_dir = tempfile.mkdtemp()

        # Protect against generator exhaustion
        datasets = list(datasets)

        # This ensures tasks are consistent for all datasets
        tasks = []
        for dataset in datasets:
            try:
                tasks.append(dataset.tasks)  # type: ignore
            except AttributeError:
                pass
        if tasks:
            task_tuples = [tuple(task_list) for task_list in tasks]
            if len(tasks) < len(datasets) or len(set(task_tuples)) > 1:
                raise ValueError('Cannot merge datasets with different task specifications')

            merge_tasks = tasks[0]
        else:
            merge_tasks = []

        def generator():
            for ind, dataset in enumerate(datasets):
                print("Merging in dataset %d/%d" % (ind, len(datasets)))
                X, y, features, ids = (dataset.X, dataset.y, dataset.features, dataset.ids)
                yield (X, y, features, ids)

        return NumpyDataset.create_dataset(generator(), data_dir=merge_dir, tasks=merge_tasks)

    def from_numpy(X: np.ndarray,
                   y: Optional[np.ndarray] = None,
                   features: Optional[np.ndarray] = None,
                   ids: Optional[np.ndarray] = None,
                   tasks: Optional[Sequence] = None,
                   data_dir: Optional[str] = None) -> "NumpyDataset":
        """Creates a DiskDataset object from specified Numpy arrays.
        Parameters
        ----------
        X: np.ndarray
          Samples array.
        y: np.ndarray, optional (default None)
          Labels array.
        features: np.ndarray, optional (default None)
          Features array.
        ids: np.ndarray, optional (default None)
          Identifiers array.
        tasks: Sequence, optional (default None)
          Tasks in this dataset
        data_dir: str, optional (default None)
          The directory to write this dataset to. If none is specified, will use
          a temporary directory instead.
        Returns
        -------
        DiskDataset
          A new `DiskDataset` constructed from the provided information.
        """
        # To unify shape handling so from_numpy behaves like NumpyDataset, we just
        # make a NumpyDataset under the hood
        dataset = NumpyDataset(X, y, features, ids)
        if tasks is None:
            tasks = dataset.get_task_names()

        # raw_data = (X, y, w, ids)
        return NumpyDataset.create_dataset(
            [(dataset.X, dataset.y, dataset.features, dataset.ids)],
            data_dir=data_dir,
            tasks=tasks)



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

        self.features2keep = None


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


    def removeElements(self, indexes):
        self.X = np.delete(self.X, indexes)
        self.y = np.delete(self.y, indexes)
        self.ids = np.delete(self.ids, indexes)
        self.features = np.delete(self.features, indexes)

    #TODO: implemtent this method also in the other subclasses
    def removeNAs(self):
        """Remove samples with NAs from the Dataset"""
        j = 0
        indexes = []
        for i in self.features:
            if len(i.shape)==0:
                indexes.append(j)
            j+=1
        print('Elements with indexes: ', indexes, ' were removed due to the presence of NAs!')
        print('The elements in question are: ', self.X[indexes])
        self.removeElements(indexes)

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



    def merge(datasets: Sequence[Dataset]) -> 'NumpyDataset':
        """Merge multiple NumpyDatasets.
        Parameters
        ----------
        datasets: List[Dataset]
          List of datasets to merge.
        Returns
        -------
        NumpyDataset
          A single NumpyDataset containing all the samples from all datasets.
        """
        X, y, features, ids = datasets[0].X, datasets[0].y, datasets[0].features, datasets[0].ids
        for dataset in datasets[1:]:
            X = np.concatenate([X, dataset.X], axis=0)
            y = np.concatenate([y, dataset.y], axis=0)
            features = np.concatenate([features, dataset.features], axis=0)
            ids = np.concatenate(
                [ids, dataset.ids],
                axis=0,
            )

        return NumpyDataset(X, y, features, ids, n_tasks=y.shape[1])
