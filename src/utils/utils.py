import pandas as pd
import joblib
import os
from typing import Any, cast, IO
import gzip
import pickle
import numpy as np


def load_pickle_file(input_file: str) -> Any:
    """Load from single, possibly gzipped, pickle file.
    Parameters
    ----------
    input_file: str
      The filename of pickle file. This function can load from
      gzipped pickle file like `XXXX.pkl.gz`.
    Returns
    -------
    Any
      The object which is loaded from the pickle file.
    """

    if ".gz" in input_file:
        with gzip.open(input_file, "rb") as unzipped_file:
            return pickle.load(cast(IO[bytes], unzipped_file))
    else:
        with open(input_file, "rb") as opened_file:
            return pickle.load(opened_file)


def save_to_disk(dataset: Any, filename: str, compress: int = 3):
    """Save a dataset to file.
    Parameters
    ----------
    dataset: str
      A data saved
    filename: str
      Path to save data.
    compress: int, default 3
      The compress option when dumping joblib file.
  """
    if filename.endswith('.joblib'):
        joblib.dump(dataset, filename, compress=compress)
    elif filename.endswith('.npy'):
        np.save(filename, dataset)
    else:
        raise ValueError("Filename with unsupported extension: %s" % filename)


def load_from_disk(filename: str) -> Any:
    """Load a dataset from file.
    Parameters
    ----------
    filename: str
      A filename you want to load data.
    Returns
    -------
    Any
      A loaded object from file.
    """

    name = filename
    if os.path.splitext(name)[1] == ".gz":
        name = os.path.splitext(name)[0]
    extension = os.path.splitext(name)[1]
    if extension == ".pkl":
        return load_pickle_file(filename)
    elif extension == ".joblib":
        return joblib.load(filename)
    elif extension == ".csv":
        # First line of user-specified CSV *must* be header.
        df = pd.read_csv(filename, header=0)
        df = df.replace(np.nan, str(""), regex=True)
        return df
    elif extension == ".npy":
        return np.load(filename, allow_pickle=True)
    else:
        raise ValueError("Unrecognized filetype for %s" % filename)


def normalize_labels_shape(y_pred):
    """Function to transform output from predict_proba (prob(0) prob(1))
    to predict format (0 or 1).
    Parameters
    ----------
    y_pred: array
      array with predictions
    Returns
    -------
    labels
      Array of predictions in the predict format (0 or 1).
    """
    labels = []
    for i in y_pred:
        if len(i) == 2:
            if i[0] > i[1]:
                labels.append(0)
            else:
                labels.append(1)
        if len(i) == 1:
            print(i)
            labels.append(int(round(i[0])))
    return np.array(labels)


'''author: Bruno Pereira
date: 28/04/2021
'''

from deepchem.trans import DAGTransformer, IRVTransformer
from deepchem.data import NumpyDataset
from Datasets.Datasets import Dataset

def dag_transformation(dataset: Dataset, max_atoms: int = 10):
    '''Function to transform ConvMol adjacency lists to DAG calculation orders.
    Adapted from deepchem'''
    new_dataset = NumpyDataset(
        X=dataset.X,
        y=dataset.y,
        ids=dataset.mols)

    transformer = DAGTransformer(max_atoms=max_atoms)
    res = transformer.transform(new_dataset)
    dataset.mols = res.ids
    dataset.X = res.X
    dataset.y = res.y

    return dataset


def irv_transformation(dataset: Dataset, K: int = 10, n_tasks: int = 1):
    '''Function to transfrom ECFP to IRV features, used by MultitaskIRVClassifier as preprocessing step
    Adapted from deepchem'''
    try:
        dummy_y = dataset.y[:, n_tasks]
    except IndexError:
        dataset.y = np.reshape(dataset.y, (np.shape(dataset.y)[0], n_tasks))
    new_dataset = NumpyDataset(
        X=dataset.X,
        y=dataset.y,
        ids=dataset.mols)

    transformer = IRVTransformer(K, n_tasks, new_dataset)
    res = transformer.transform(new_dataset)
    dataset.mols = res.ids
    dataset.X = res.X
    dataset.y = np.reshape(res.y, (np.shape(res.y)[0],))

    return dataset