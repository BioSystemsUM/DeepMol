# SOURCE: https://github.com/deepchem/deepchem/blob/master/deepchem/data/datasets.py

import numpy as np


class Dataset(object):
    """Abstract base class for datasets
  """

    def __init__(self):
        raise NotImplementedError()

    def __len__(self):
        """Get the number of elements in the dataset."""
        return len(self.y)

    def get_shape(self):
        """Get the shape of all the elements of the dataset.
        Returns three tuples, shape of X, y and ids arrays.
        """
        return self.X.shape, self.y.shape, self.ids.shape

    def X(self):
        """Get the X vector for this dataset as a single numpy array."""
        return self.X

    def y(self):
        """Get the y vector for this dataset as a single numpy array."""
        return self.y

    def ids(self):
        """Get the ids vector for this dataset as a single numpy array."""
        return self.ids

class DeepMolDataset(Dataset):
    '''
    ...
    '''

    def __init__(self, x, y=None, ids=None, n_tasks=1):
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

        if len(x) > 0:
            if not isinstance(x, np.ndarray):
                x = np.array(x)

            if y is None:
                y = np.zeros((len(x), n_tasks), np.float32)
            else :
                if not isinstance(y, np.ndarray):
                    y = np.array(y)
        else :
            raise ValueError("Invalid or Empty Dataset!")

        self.X = x
        self.y = y
        self.ids = np.array(ids, dtype=object)
        self.n_tasks = n_tasks


