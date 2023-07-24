from typing import Union

import numpy as np
import pandas as pd

from deepmol.loggers.logger import Logger


def merge_arrays(array1: np.ndarray, size1: int, array2: np.ndarray, size2: int) -> np.ndarray:
    """
    Merges two arrays into a single array.
    If any of the arrays are None, it creates an array of the defined size with np.nan values.
    The resulting array will have size1 + size2 elements.

    Parameters
    ----------
    array1: np.ndarray
        The first array to merge.
    size1: int
        The size of the first array.
    array2: np.ndarray
        The second array to merge.
    size2: int
        The size of the second array.

    Returns
    -------
    merged: np.ndarray
        The merged array.
    """
    if array1 is None:
        array1 = [np.nan] * size1
    if array2 is None:
        array2 = [np.nan] * size2
    merged = np.append(array1, array2, axis=0)
    return merged


def merge_arrays_of_arrays(array1: np.ndarray, array2: np.ndarray) -> Union[np.ndarray, None]:
    """
    Merges two arrays of arrays into a single array of arrays.
    Both arrays must have the same shape[1] ("columns"), otherwise None is returned.

    Parameters
    ----------
    array1: np.ndarray
        The first array of arrays to merge.
    array2: np.ndarray
        The second array of arrays to merge.

    Returns
    -------
    merged: np.ndarray
        The merged array of arrays.
    """
    logger = Logger()
    if len(array1.shape) != len(array2.shape):
        logger.error('Features are not the same length/type... Recalculate features for all inputs!')
        return None
    if array1.shape[1] != array2.shape[1]:
        logger.error('Features are not the same length/type... Recalculate features for all inputs!')
        return None
    merged = np.concatenate([array1, array2], axis=0)
    return merged


def _get_n_classes(dataset):
    """
    Get the number of classes of the dataset for each task.

    Parameters
    ----------
    dataset : Dataset
        Dataset object.

    Returns
    -------
    list of int
        Number of classes for each task.
    """
    if dataset.mode == 'classification':
        n_classes = [len(set(dataset.y))]
    elif dataset.mode == 'regression':
        n_classes = [1]
    elif isinstance(dataset.mode, list):
        n_classes = []
        for i in range(len(dataset.mode)):
            if dataset.mode[i] == 'classification':
                class_set = set(dataset.y[i])
                class_set = {x for x in class_set if pd.notna(x)}
                n_classes.append(len(class_set))
            elif dataset.mode[i] == 'regression':
                n_classes.append(1)
            else:
                raise ValueError(f'Unknown mode {dataset.mode[i]}')
    else:
        raise ValueError(f'Unknown mode {dataset.mode}')
    return n_classes
