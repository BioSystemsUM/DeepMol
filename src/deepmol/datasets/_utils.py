import numpy as np


def merge_arrays(array1: np.array, size1: int, array2: np.array, size2: int):
    if array1 is None:
        array1 = [np.nan] * size1
    if array2 is None:
        array2 = [np.nan] * size2
    merged = np.append(array1, array2, axis=0)
    return merged


def merge_arrays_of_arrays(array1, array2):
    if len(array1.shape) != len(array2.shape):
        print('Features are not the same length/type... Recalculate features for all inputs!')
        return None
    if array1.shape[1] != array2.shape[1]:
        print('Features are not the same length/type... Recalculate features for all inputs!')
        return None
    merged = np.concatenate([array1, array2], axis=0)
    return merged
