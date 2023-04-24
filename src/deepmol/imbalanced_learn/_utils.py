import uuid

import numpy as np


def _get_new_ids(original_x: np.array, new_x, original_ids):
    """
    Get new ids for the new dataset.

    Parameters
    ----------
    original_x: np.array
        Original dataset.
    new_x: np.array
        Sampled dataset.
    original_ids: np.array
        Original ids.

    Returns
    -------
    np.array
        Ids of the sampled dataset.
    """
    ids = []
    for i in range(new_x.shape[0]):
        # check if the sample is artificial or not be checcking if the row is in the original dataset
        if np.any(np.all(new_x[i] == original_x, axis=1)):
            ids.append(original_ids[np.where(np.all(new_x[i] == original_x, axis=1))[0][0]])
        else:
            ids.append(f"ib_{uuid.uuid4().hex}")
    return np.array(ids)
