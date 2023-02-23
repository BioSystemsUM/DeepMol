from typing import List, Dict

from deepchem.utils import ConformerGenerator
from rdkit.Chem import Mol


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
        try:
            atoms = mol.GetNumAtoms()
            if atoms > best:
                best = atoms
        except Exception as e:
            print('Molecule with index', i, 'was not converted from SMILES into RDKIT object')
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
            print('Molecules with index', i, 'was not able to achieve a correct conformation')
            print('Appending empty list')
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
