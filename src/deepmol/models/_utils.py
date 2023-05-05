import os
import pickle
from typing import Any

import joblib
import numpy as np

from deepmol.datasets import Dataset
from deepmol.splitters.splitters import Splitter, SingletaskStratifiedSplitter, RandomSplitter
from deepmol.utils.utils import load_pickle_file


# TODO: review this function
def save_to_disk(model: 'Model', filename: str, compress: int = 3):
    """
    Save a model to a file.

    Parameters
    ----------
    model: Model
        The model you want to save.
    filename: str
        Path to save data.
    compress: int, default 3
        The compress option when dumping joblib file.
  """
    if filename.endswith('.joblib'):
        joblib.dump(model, filename, compress=compress)
    elif filename.endswith('.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError("Filename with unsupported extension: %s" % filename)


def load_model_from_disk(filename: str) -> Any:
    """
    Load model from file.

    Parameters
    ----------
    filename: str
        A filename you want to load.

    Returns
    -------
    Any
      A loaded object from file.
    """
    name = filename
    extension = os.path.splitext(name)[1]
    if extension == ".pkl":
        return load_pickle_file(filename)
    elif extension == ".joblib":
        return joblib.load(filename)
    else:
        raise ValueError("Unrecognized filetype for %s" % filename)


def _get_splitter(dataset: Dataset) -> Splitter:
    """
    Returns a splitter for a dataset.
    """
    if dataset.mode == 'classification' and dataset.n_tasks == 1:
        splitter = SingletaskStratifiedSplitter()
    elif dataset.mode == 'regression':
        splitter = RandomSplitter()
    else:
        splitter = RandomSplitter()
    return splitter
