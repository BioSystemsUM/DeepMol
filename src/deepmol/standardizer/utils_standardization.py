from copy import deepcopy

from rdkit.Chem import Mol, RemoveStereochemistry, GetMolFrags, AddHs, RemoveHs, Kekulize
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.MolStandardize import rdMolStandardize


def remove_isotope_info(mol: Mol) -> Mol:
    """
    Remove isotope information from a molecule.

    Parameters
    ----------
    mol: Mol
        RDKit Mol object

    Returns
    ----------
    mol: Mol
        molecule without isotope insformation
    """
    mol = deepcopy(mol)
    for atm in mol.GetAtoms():
        atm.SetIsotope(0)
    if not mol:
        raise ValueError('Failed to remove isotope information!')
    return mol


def uncharge(mol: Mol) -> Mol:
    """
    Neutralize a molecule.

    Parameters
    ----------
    mol: Mol
        RDKit Mol object

    Returns
    ----------
    mol: Mol
        neutralized molecule
    """
    mol = deepcopy(mol)
    uncharger = rdMolStandardize.Uncharger()
    mol = uncharger.uncharge(mol)
    if not mol:
        raise ValueError('Failed to neutralize molecule!')
    return mol


def remove_stereo(mol: Mol) -> Mol:
    """
    Remove stereochemistry from a molecule.

    Parameters
    ----------
    mol: Mol
        RDKit Mol object

    Returns
    ----------
    mol: Mol
        molecule without stereochemistry
    """
    mol = deepcopy(mol)
    RemoveStereochemistry(mol)
    if not mol:
        raise ValueError('Failed to remove stereochemistry from the molecule!')
    return mol


def count_non_hs_atoms(mol: Mol) -> int:
    """
    Count the number of atoms in a molecule excluding Hs.

    Parameters
    ----------
    mol: Mol
        RDKit Mol object

    Returns
    ----------
    ans: int
        number of non Hs atoms in the molecule.
    """
    ans = 0
    for atm in mol.GetAtoms():
        if atm.GetAtomicNum() != 1:
            ans += 1
    return ans


def keep_biggest(mol: Mol) -> Mol:
    """
    Keep only the biggest fragment according to number of non H atoms or molecular weight if tied.

    Parameters
    ----------
    mol: Mol
        RDKit Mol object

    Returns
    ----------
    mol: Mol
        biggest fragment of the molecule
    """
    molfrags = GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    mol_out = deepcopy(mol)
    if len(molfrags) > 1:
        accepted_nbr_atm = 0
        accepted_mw = 0
        for f in molfrags:
            nbr_atm = count_non_hs_atoms(f)
            if nbr_atm > accepted_nbr_atm or (nbr_atm == accepted_nbr_atm and MolWt(f) > accepted_mw):
                accepted_nbr_atm = nbr_atm
                accepted_mw = MolWt(f)
                mol_out = f  # keep only the biggest fragment
    return mol_out


def add_hydrogens(mol: Mol, addCoords: bool = True) -> Mol:
    """
    Explicit all hydrogens.

    Parameters
    ----------
    mol: Mol
        RDKit Mol object
    addCoords: bool
        Add coordinate to added Hs

    Returns
    ----------
    mol: Mol
        molecule with all Hs explicit.
    """
    return AddHs(mol, explicitOnly=False, addCoords=addCoords)


def remove_hydrogens(mol: Mol, addCoords: bool = True) -> Mol:
    """
    Implicit all hydrogens.

    Parameters
    ----------
    mol: Mol
        RDKit Mol object
    addCoords: bool
        Add coordinate to added Hs

    Returns
    ----------
    mol: Mol
        molecule with all Hs implicit.
    """
    return RemoveHs(mol, explicitOnly=False, addCoords=addCoords)


def kekulize(mol: Mol) -> Mol:
    """
    Kekulize compound.

    Parameters
    ----------
    mol: Mol
        RDKit Mol object

    Returns
    ----------
    mol: Mol
        kekulized molecule
    """
    mol = deepcopy(mol)
    Kekulize(mol, clearAromaticFlags=True)
    return mol
