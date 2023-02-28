from typing import Union

import numpy as np

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


def check_values(values: Union[np.ndarray, float, int]) -> bool:
    """
    Checks if the array is empty or if contains any np.nan values.

    Parameters
    ----------
    values: Union[np.ndarray, float, int]
        The values to check.

    Returns
    -------
    bool
        True if the array is empty or contains np.nan values, False otherwise.
    """
    if values is None:
        return True
    if isinstance(values, np.ndarray):
        if values.size == 0:
            return True
        if isinstance(values[0], float) or isinstance(values[0], int):
            if np.isnan(np.dot(values, values)):
                return True
            return False
    if isinstance(values, float) or isinstance(values, int):
        if np.isnan(values):
            return True
        return False
    return False
