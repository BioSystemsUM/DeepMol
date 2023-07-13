import pandas as pd
import joblib
import os
from typing import Any, cast, IO, List, Union
import gzip
import pickle
import numpy as np

from rdkit.Chem import Mol, rdmolfiles, rdmolops

from rdkit import Chem


def get_class(name: str) -> object:
    """
    Get a class from a string.

    Parameters
    ----------
    name: str
        A class name you want to get.

    Returns
    -------
    object
        A class object.
    """
    components = name.split(".")
    mod = __import__(".".join(components[:-1]), fromlist=[components[-1]])
    return getattr(mod, components[-1])


def smiles_to_mol(smiles: str, **kwargs) -> Union[Mol, None]:
    """
    Convert SMILES to RDKit molecule object.
    Parameters
    ----------
    smiles: str
        SMILES string to convert.
   kwargs:
           Keyword arguments for `rdkit.Chem.MolFromSmiles`.
    Returns
    -------
    Mol
        RDKit molecule object.
    """
    try:
        return Chem.MolFromSmiles(smiles, **kwargs)
    except TypeError:
        return None


def mol_to_smiles(mol: Mol, **kwargs) -> Union[str, None]:
    """
    Convert SMILES to RDKit molecule object.
    Parameters
    ----------
    mol: Mol
        RDKit molecule object to convert.
   kwargs:
           Keyword arguments for `rdkit.Chem.MolToSmiles`.
    Returns
    -------
    smiles: str
        SMILES string.
    """
    try:
        return Chem.MolToSmiles(mol, **kwargs)
    except TypeError:
        return None


def canonicalize_mol_object(mol_object: Mol) -> Mol:
    """
    Canonicalize a molecule object.

    Parameters
    ----------
    mol_object: Mol
        Molecule object to canonicalize.

    Returns
    -------
    Mol
        Canonicalized molecule object.
    """
    try:
        # SMILES is unique, so set a canonical order of atoms
        new_order = rdmolfiles.CanonicalRankAtoms(mol_object)
        mol_object = rdmolops.RenumberAtoms(mol_object, new_order)
    except Exception as e:
        mol_object = mol_object

    return mol_object


def load_pickle_file(input_file: str) -> Any:
    """
    Load from single, possibly gzipped, pickle file.

    Parameters
    ----------
    input_file: str
        The filename of pickle file. This function can load from gzipped pickle file like `XXXX.pkl.gz`.

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


def load_from_disk(filename: str) -> Any:
    """
    Load object from file.

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


def normalize_labels_shape(y_pred: Union[List, np.ndarray], n_tasks: int) -> np.ndarray:
    """
    Function to transform output from predict_proba (prob(0) prob(1)) to predict format (0 or 1).

    Parameters
    ----------
    y_pred: array
        array with predictions
    n_tasks: int
        number of tasks

    Returns
    -------
    labels
        Array of predictions in the format [0, 1, 0, ...]/[[0, 1, 0, ...], [0, 1, 1, ...], ...]
    """
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    if n_tasks == 1:
        labels = _normalize_singletask_labels_shape(y_pred)
    else:
        if len(y_pred.shape) == 3:
            if y_pred.shape[2] > 1:
                y_pred = np.array([np.array([j[1] for j in i]) for i in y_pred])
            else:
                y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        labels = []
        for task in y_pred:
            labels.append(_normalize_singletask_labels_shape(task))

    labels = np.array(labels)
    return labels


def _normalize_singletask_labels_shape(y_pred: Union[List, np.ndarray]) -> np.ndarray:
    """
    Function to transform output from predict_proba (prob(0) prob(1)) to predict format (0 or 1).

    Parameters
    ----------
    y_pred: array
        array with predictions

    Returns
    -------
    labels
        Array of predictions in the format [0, 1, 0, ...]/[[0, 1, 0, ...], [0, 1, 1, ...], ...]
    """
    labels = []
    # list of probabilities in the format [0.1, 0.9, 0.2, ...]
    if isinstance(y_pred[0], (np.floating, float)):
        return np.array(y_pred)
    elif isinstance(y_pred[0], (np.integer, int)):
        return np.array(y_pred)
    # list of lists of probabilities in the format [[0.1], [0.2], ...]
    elif len(y_pred[0]) == 1:
        return np.array([i[0] for i in y_pred])
    # list of lists of probabilities in the format [[0.1, 0.9], [0.2, 0.8], ...]
    elif len(y_pred[0]) == 2:
        return np.array([i[1] for i in y_pred])
    elif len(y_pred[0]) > 2:
        return np.array([np.argmax(i) for i in y_pred])
    else:
        raise ValueError("Unknown format for y_pred!")
