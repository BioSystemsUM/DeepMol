import uuid

import numpy as np


def _get_new_ids_smiles_mols(original_x: np.array,
                             new_x: np.array,
                             original_ids: np.array,
                             original_smiles: np.array,
                             original_mols: np.array) -> (np.array, np.array, np.array):
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
    original_smiles: np.array
        Original smiles.
    original_mols: np.array
        Original mols.

    Returns
    -------
    ids: np.array
        Ids of the sampled dataset.
    smiles: np.array
        Smiles of the sampled dataset.
    mols: np.array
        Mols of the sampled dataset.
    """
    ids = []
    smiles = []
    mols = []
    assigned_ids = set()
    for i in range(new_x.shape[0]):
        # check if the sample is artificial or not by checking if the row is in the original dataset
        if np.any(np.all(new_x[i] == original_x, axis=1)):
            idx = np.where(np.all(new_x[i] == original_x, axis=1))[0][0]
            id_ = original_ids[idx]
            # check if id has already been assigned to another sample
            if id_ in assigned_ids:
                id_ = f"ib_{id_}"
            assigned_ids.add(id_)
            ids.append(id_)
            smiles.append(original_smiles[idx])
            mols.append(original_mols[idx])
        else:
            id_ = f"ib_{uuid.uuid4().hex}"
            # check if id has already been assigned to another sample
            while id_ in assigned_ids:
                id_ = f"ib_{uuid.uuid4().hex}"
            assigned_ids.add(id_)
            ids.append(id_)
            smiles.append(None)
            mols.append(None)
    return np.array(ids), np.array(smiles), np.array(mols)
