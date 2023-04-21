import joblib
import numpy as np

from deepmol.datasets import Dataset
from deepmol.splitters.splitters import Splitter, SingletaskStratifiedSplitter, RandomSplitter


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
    elif filename.endswith('.npy'):
        np.save(filename, model)
    else:
        raise ValueError("Filename with unsupported extension: %s" % filename)


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
