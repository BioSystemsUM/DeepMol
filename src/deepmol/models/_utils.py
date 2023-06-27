import os
import pickle
from typing import Any, Union

import dill
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
        try:
            with open(filename, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (TypeError, AttributeError):
            # dump with dill
            with open(filename, 'wb') as f:
                dill.dump(model, f)
    else:
        raise ValueError("Filename with unsupported extension: %s" % filename)


def load_from_disk(filename: str) -> Any:
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
        try:
            return load_pickle_file(filename)
        except (TypeError, AttributeError):
            with open(filename, 'rb') as f:
                return dill.load(f)
    elif extension == ".joblib":
        return joblib.load(filename)
    else:
        raise ValueError("Unrecognized filetype for %s" % filename)


def _get_splitter(dataset: Dataset) -> Splitter:
    """
    Returns a splitter for a dataset.

    Parameters
    ----------
    dataset: Dataset
        The dataset to get the splitter for.
    """
    if dataset.mode == 'classification' and dataset.n_tasks == 1:
        splitter = SingletaskStratifiedSplitter()
    elif dataset.mode == 'regression':
        splitter = RandomSplitter()
    else:
        splitter = RandomSplitter()
    return splitter


def _save_keras_model(file_path: str,
                      model: Union['deepchem.models.KerasModel', 'deepmol.models.keras_model.KerasModel'],
                      parameters_to_save: dict,
                      model_builder: callable = None):
    """
    Saves a keras model to disk.

    Parameters
    ----------
    file_path: str
        The path to save the model to.
    model: KerasModel
        The keras model.
    parameters_to_save: dict
        The parameters to save.
    model_builder: callable
        The model builder.
    """
    os.makedirs(file_path, exist_ok=True)
    if model_builder is not None:
        file_path_model_builder = os.path.join(file_path, 'model_builder.pkl')
        save_to_disk(model_builder, file_path_model_builder)
    # write model in h5 format
    model_file_path = os.path.join(file_path, 'model.h5')
    model.save(model_file_path)
    # write parameters in pickle format
    file_path_parameters = os.path.join(file_path, 'model_parameters.pkl')
    save_to_disk(parameters_to_save, file_path_parameters)


def multi_label_binarize(y_pred_proba, threshold=0.5) -> np.ndarray:
    """
    Binarize the predicted probabilities for multi-label classification.

    Parameters
    ----------
    y_pred_proba: np.ndarray
        Predicted probabilities with shape (n_samples, n_classes).
    threshold: float
        Threshold for binarization.

    Returns
    -------
        np.ndarray: Binary predictions with shape (n_samples, n_classes).
    """
    n_samples, n_classes = y_pred_proba.shape
    y_pred = np.zeros((n_samples, n_classes))
    for i in range(n_classes):
        y_pred[:, i] = (y_pred_proba[:, i] >= threshold).astype(int)
    return y_pred


def get_prediction_from_proba(dataset: Dataset, y_pred_proba: np.ndarray) -> np.ndarray:
    """
    Get predictions from predicted probabilities.

    Parameters
    ----------
    dataset: Dataset
        The dataset.
    y_pred_proba: np.ndarray
        Predicted probabilities with shape (n_samples, n_classes).

    Returns
    -------
    np.ndarray: Predictions with shape (n_samples,).

    """
    if dataset.mode == "classification" or dataset.mode == "multitask":
        if np.all((dataset.y == 0) | (dataset.y == 1)):
            if y_pred_proba.shape[1] == 1:
                y_pred = np.array([1 if pred >= 0.5 else 0 for pred in y_pred_proba])
            else:
                y_pred = multi_label_binarize(y_pred_proba)
        else:
            y_pred = []
            if not len(y_pred_proba) == 0:
                y_pred = np.argmax(y_pred_proba, axis=1)
                return y_pred
    elif dataset.mode == "regression":
        y_pred = y_pred_proba
    else:
        y_pred = []
        if not len(y_pred_proba) == 0:
            y_pred = np.argmax(y_pred_proba, axis=1)
            return y_pred

    return y_pred
