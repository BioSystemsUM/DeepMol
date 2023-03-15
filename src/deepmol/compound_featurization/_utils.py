from typing import List, Dict, Tuple

import numpy as np
from deepchem.utils import ConformerGenerator
from rdkit import DataStructs
from rdkit.Chem import Mol, AllChem

from deepmol.loggers import Logger


def find_maximum_number_atoms(molecules: List[Mol]) -> int:
    """
    Finds maximum number of atoms within a set of molecules

    Parameters
    ----------
    molecules: List[Mol]
        List of rdkit mol objects

    Returns
    -------
    best: int
        Maximum number of atoms in a molecule in the list.
    """
    best = 0
    for i, mol in enumerate(molecules):
        atoms = mol.GetNumAtoms()
        if atoms > best:
            best = atoms
    return best


def get_conformers(molecules: List[Mol], generator: ConformerGenerator) -> List[Mol]:
    """
    Gets conformers for molecules with a specific generator

    Parameters
    ----------
    molecules: List[Mol]
        List of rdkit mol objects
    generator: ConformerGenerator
        DeepChem conformer generator.

    Returns
    -------
    new_conformations: List[Mol]
        List of rdkit mol objects with conformers.
    """
    new_conformations = []
    for i, mol in enumerate(molecules):
        try:
            conf = generator.generate_conformers(mol)
            new_conformations.append(conf)
        except Exception as e:
            logger = Logger()
            logger.error(f"Could not generate conformers for molecule {i} with error {e}")
            new_conformations.append([])
    return new_conformations


def get_dictionary_from_smiles(smiles: List[str], max_len: int) -> Dict[str, int]:
    """
    Dictionary of character to index mapping
    Adapted from deepchem.

    Parameters
    ----------
    smiles: List[str]
        List of SMILES string
    max_len: int
        Maximum length of SMILES string

    Returns
    -------
    dictionary: Dict[str, int]
        Dictionary of character to index mapping
    """

    pad_token = "<pad>"
    out_of_vocab_token = "<unk>"

    char_set = set()
    for smile in smiles:
        if len(smile) <= max_len:
            char_set.update(set(smile))

    unique_char_list = list(char_set) + [pad_token, out_of_vocab_token]
    dictionary = {letter: idx for idx, letter in enumerate(unique_char_list)}
    return dictionary


def calc_morgan_fingerprints(mols: np.ndarray, **kwargs) -> List[np.ndarray]:
    """
    Calculates Morgan fingerprints for a array of molecules.

    Parameters
    ----------
    mols: np.ndarray
        Array of rdkit mol objects
    kwargs:
        Keyword arguments for the Morgan fingerprint calculation.

    Returns
    -------
    fps: List[np.ndarray]
        List of Morgan fingerprints
    """
    if "radius" not in kwargs:
        kwargs["radius"] = 2
    if "nBits" not in kwargs:
        kwargs["nBits"] = 1024
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, **kwargs) for m in mols]
    return fps


def calc_similarity(first_fp_idx: int, second_fp_idx: int,  fps: List[np.ndarray]) -> Tuple[int, int, float]:
    """
    Calculates the Tanimoto similarity between two fingerprints.

    Parameters
    ----------
    first_fp_idx: int
        Index of the first fingerprint
    second_fp_idx: int
        Index of the second fingerprint
    fps: List[np.ndarray]
        List of Morgan fingerprints

    Returns
    -------
    i: int
        Index of the first fingerprint
    j: int
        Index of the second fingerprint
    similarity: float
        Tanimoto similarity between the two fingerprints
    """
    return first_fp_idx, second_fp_idx, DataStructs.TanimotoSimilarity(fps[first_fp_idx], fps[second_fp_idx])
