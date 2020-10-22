
import numpy as np
import pandas as pd

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

    #chunk_num = 1
    if chunk_size is None:
        if keep_fields is None:
            return pd.read_csv(input_file)
        else : return pd.read_csv(input_file)[keep_fields]
    else:
        for df in pd.read_csv(input_file, chunksize=chunk_size):
            #df = df.replace(np.nan, str(""), regex=True)
            #chunk_num += 1
            if keep_fields is None:
                return df
            else: return df[keep_fields]

class Dataset(object):
    """Abstract base class for datasets
    """

    def __init__(self):
        raise NotImplementedError()

    def __len__(self):
        """Get the number of elements in the dataset."""
        raise NotImplementedError()

    def get_shape(self):
        """Get the shape of all the elements of the dataset.
        Returns two tuples, shape of X (number of examples) and y (number of tasks).
        """
        raise NotImplementedError()

    def X(self):
        """Get the X (number of examples) vector for this dataset as a single numpy array."""
        raise NotImplementedError()

    def y(self):
        """Get the y (number of tasks) vector for this dataset as a single numpy array."""
        raise NotImplementedError()

class CSVLoader(Dataset):
    '''
    ...
    '''

    def __init__(self, dataset_path, input_field, output_fields=None,
                 id_field=None, user_features=None, keep_all_fields=False, chunk_size=None):
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

        #TODO: provide user features in a better way? Maybe give the columns that do not contain the features?
        if not isinstance(user_features, list) and user_features is not None:
            raise ValueError("User features must be a list of string containing "
                             "the features fields or None.")

        self.dataset_path = dataset_path
        self.tasks = output_fields
        self.input_field = input_field

        if id_field is None:
            self.id_field = input_field
        else:
            self.id_field = id_field

        self.user_features = user_features

        if keep_all_fields:
            self.dataset = self._get_dataset(dataset_path, keep_fields=None, chunk_size=chunk_size)
        else:
            columns = [id_field, input_field]
            for field in output_fields:
                columns.append(field)
            if user_features is None:
                self.dataset = self._get_dataset(dataset_path, keep_fields=columns, chunk_size=chunk_size)
            else:
                self.dataset = self._get_dataset(dataset_path, keep_fields=columns, chunk_size=chunk_size)

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
        Returns four tuples, giving the shape of the X, y, w, and ids arrays.
        """
        return self.X.shape, self.y.shape, self.ids.shape

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

    def _get_dataset(self, dataset_path, keep_fields=None, chunk_size=None):
        """Defines a generator which returns data for each shard
        Parameters
        ----------
        input_files: List[str]
          List of filenames to process
        chunk_size: int, optional
          The size of a shard of data to process at a time.
        Returns
        -------
        Iterator[pd.DataFrame]
          Iterator over chunks
        """
        return load_csv_file(dataset_path, keep_fields, chunk_size)
