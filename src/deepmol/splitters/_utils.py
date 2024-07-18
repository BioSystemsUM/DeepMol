import math
from typing import List

import numpy as np
from rdkit.Chem import AllChem

from deepmol.datasets import Dataset


def get_train_valid_test_indexes(scaffold_sets: List[List[int]],
                                 train_cutoff: float,
                                 test_cutoff: float,
                                 valid_cutoff: float,
                                 frac_train: float,
                                 frac_test: float,
                                 homogenous_datasets: bool):
    """
    Get the indexes of the train, valid and test sets based on the scaffold sets.

    Parameters
    ----------
    scaffold_sets: List[List[int]]
        The scaffold sets.
    train_cutoff: float
        The cutoff to define the train set.
    test_cutoff: float
        The cutoff to define the test set.
    valid_cutoff: float
        The cutoff to define the valid set.
    frac_train: float
        The fraction of the train set.
    frac_test: float
        The fraction of the test set.
    homogenous_datasets: bool
        If True, the train, valid and test sets will be homogenous.

    Returns
    -------
    Tuple[List[int], List[int], List[int]]:
        The indexes of the train, valid and test sets.
    """
    train_inds = np.array([])
    valid_inds = np.array([])
    test_inds = np.array([])

    for scaffold_set in scaffold_sets:
        scaffold_set = np.array(scaffold_set)
        if not homogenous_datasets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(test_inds) + len(scaffold_set) <= test_cutoff:
                    test_inds = np.hstack([test_inds, scaffold_set])
                elif len(valid_inds) + len(scaffold_set) <= valid_cutoff:
                    valid_inds = np.hstack([valid_inds, scaffold_set])
                else:
                    np.random.shuffle(scaffold_set)
                    train_index = int(np.round(len(scaffold_set) * frac_train))
                    test_index = int(np.round(len(scaffold_set) * frac_test)) + train_index
                    train_inds = np.hstack([train_inds, scaffold_set[:train_index]])
                    test_inds = np.hstack([test_inds, scaffold_set[train_index:test_index]])
                    valid_inds = np.hstack([valid_inds, scaffold_set[test_index:]])
            else:
                train_inds = np.hstack([train_inds, scaffold_set])

        else:
            np.random.shuffle(scaffold_set)
            if len(train_inds) >= train_cutoff:
                if len(test_inds) >= test_cutoff:
                    valid_inds = np.hstack([valid_inds, scaffold_set])
                else:
                    test_inds = np.hstack([test_inds, scaffold_set])
            else:
                # to avoid leaks to valid_inds when valid_frac=0
                train_index = math.floor(len(scaffold_set) * frac_train + 0.5)
                test_index = int(np.round(len(scaffold_set) * frac_test)) + train_index
                train_inds = np.hstack([train_inds, scaffold_set[:train_index]])
                test_inds = np.hstack([test_inds, scaffold_set[train_index:test_index]])
                valid_inds = np.hstack([valid_inds, scaffold_set[test_index:]])
    return list(map(int, train_inds)), list(map(int, valid_inds)), list(map(int, test_inds))


def get_mols_for_each_class(dataset: Dataset):
    """
    Get the molecules for each class.

    Parameters
    ----------
    dataset: Dataset
        The dataset.

    Returns
    -------
    Tuple[Dict[int, List[Mol]], Dict[int, List[int]]]:
        The molecules to class map and the indices to class map.
    """
    mols_classes_map = {}
    indices_classes_map = {}
    for i, mol in enumerate(dataset.mols):

        if dataset.y[i] not in mols_classes_map:
            mols_classes_map[dataset.y[i]] = [mol]
            indices_classes_map[dataset.y[i]] = [i]

        else:
            mols_classes_map[dataset.y[i]].append(mol)
            indices_classes_map[dataset.y[i]].append(i)

    return mols_classes_map, indices_classes_map


def get_fingerprints_for_each_class(dataset: Dataset):
    """
    Get the fingerprints for each class.

    Parameters
    ----------
    dataset: Dataset
        The dataset.

    Returns
    -------
    Tuple[Dict[int, List[DataStructs.ExplicitBitVect]], Dict[int, List[int]], List[DataStructs.ExplicitBitVect]]:
        The fingerprints to class map, the indices to class map and the complete set of fingerprints.
    """
    fps_classes_map = {}
    indices_classes_map = {}
    all_fps = []
    classes = dataset.y is not None
    for i, mol in enumerate(dataset.mols):

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
        all_fps.append(fp)
        if classes:
            if dataset.y[i] not in fps_classes_map:
                fps_classes_map[dataset.y[i]] = [fp]
                indices_classes_map[dataset.y[i]] = [i]

            else:
                fps_classes_map[dataset.y[i]].append(fp)
                indices_classes_map[dataset.y[i]].append(i)
        else:
            if 0 not in fps_classes_map:
                fps_classes_map[0] = [fp]
                indices_classes_map[0] = [i]

            else:
                fps_classes_map[0].append(fp)
                indices_classes_map[0].append(i)

    return fps_classes_map, indices_classes_map, all_fps
