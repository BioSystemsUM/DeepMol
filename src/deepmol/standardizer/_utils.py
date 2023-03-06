from copy import deepcopy

from rdkit.Chem import Mol, RemoveStereochemistry, GetMolFrags, AddHs, RemoveHs, Kekulize, SanitizeMol, SanitizeFlags, \
    Cleanup, AssignStereochemistry
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.MolStandardize import rdMolStandardize


def basic_standardizer(mol: Mol) -> Mol:
    """
    Basic standardization of a molecule.
    Performs only Sanitization (Kekulize, check valencies, set aromaticity, conjugation and hybridization).

    Parameters
    ----------
    mol: Mol
        RDKit Mol object

    Returns
    -------
    mol: Mol
        Standardized mol.
    """
    SanitizeMol(mol)
    return mol


# Customizable standardizer and custom parameters
# adapted from https://github.com/brsynth/RetroPathRL
simple_standardisation = {
    'REMOVE_ISOTOPE': False,
    'NEUTRALISE_CHARGE': False,
    'REMOVE_STEREO': True,
    'KEEP_BIGGEST': False,
    'ADD_HYDROGEN': False,
    'KEKULIZE': False,
    'NEUTRALISE_CHARGE_LATE': True}

heavy_standardisation = {
    'REMOVE_ISOTOPE': True,
    'NEUTRALISE_CHARGE': True,
    'REMOVE_STEREO': True,
    'KEEP_BIGGEST': True,
    'ADD_HYDROGEN': True,
    'KEKULIZE': False,
    'NEUTRALISE_CHARGE_LATE': True}


def custom_standardizer(mol: Mol,
                        REMOVE_ISOTOPE: bool = True,
                        NEUTRALISE_CHARGE: bool = True,
                        REMOVE_STEREO: bool = False,
                        KEEP_BIGGEST: bool = True,
                        ADD_HYDROGEN: bool = True,
                        KEKULIZE: bool = True,
                        NEUTRALISE_CHARGE_LATE: bool = True) -> Mol:
    """
    Tunable sequence of filters for standardization.

    Operations will be made in the following order:
     1 RDKit Cleanup      -- always
     2 RDKIT SanitizeMol  -- always
     3 Remove isotope     -- optional (default: True)
     4 Neutralise charges -- optional (default: True)
     5 RDKit SanitizeMol  -- if 3 or 4
     6 Remove stereo      -- optional (default: False)
     7 Keep biggest       -- optional (default: True)
     8 RDKit SanitizeMol  -- if any (6, 7)
     9 Add hydrogens      -- optional (default: True)
    10 Kekulize           -- optional (default: True)

    Parameters
    ----------
    mol: Mol
        RDKit Mol object
    REMOVE_ISOTOPE: bool
        Remove isotope information from the molecule.
    NEUTRALISE_CHARGE: bool
        Neutralise the charge of the molecule.
    REMOVE_STEREO: bool
        Remove stereo information from the molecule.
    KEEP_BIGGEST: bool
        Keep only the biggest fragment of the molecule.
    ADD_HYDROGEN: bool
        Add explicit hydrogens to the molecule.
    KEKULIZE: bool
        Kekulize the molecule.
    NEUTRALISE_CHARGE_LATE: bool
        Neutralise the charge of the molecule after sanitization.

    Returns
    -------
    mol: Mol
        Standardized mol.
    """

    Cleanup(mol)
    SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL, catchErrors=False)
    AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)

    if REMOVE_ISOTOPE:
        mol = remove_isotope_info(mol)
    if NEUTRALISE_CHARGE:
        mol = uncharge(mol)
    if any([REMOVE_ISOTOPE, NEUTRALISE_CHARGE]):
        SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL, catchErrors=False)
    if REMOVE_STEREO:
        mol = remove_stereo(mol)
    if KEEP_BIGGEST:
        mol = keep_biggest(mol)
    if any([REMOVE_STEREO, KEEP_BIGGEST]):
        SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL, catchErrors=False)
    if NEUTRALISE_CHARGE_LATE:
        mol = uncharge(mol)
        SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL, catchErrors=False)
    if ADD_HYDROGEN:
        mol = add_hydrogens(mol, addCoords=True)
    else:
        mol = remove_hydrogens(mol)
    if KEKULIZE:
        mol = kekulize(mol)
    return mol


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


def remove_hydrogens(mol: Mol) -> Mol:
    """
    Implicit all hydrogens.

    Parameters
    ----------
    mol: Mol
        RDKit Mol object

    Returns
    ----------
    mol: Mol
        molecule with all Hs implicit.
    """
    return RemoveHs(mol, implicitOnly=False)


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
