from rdkit.Chem import Cleanup, SanitizeMol, SanitizeFlags
from rdkit.Chem.AllChem import AssignStereochemistry
from standardizer.utils_standardization import remove_isotope_info, uncharge, remove_stereo, kekulize, \
    keep_biggest, add_hydrogens, remove_hydrogens

from standardizer.MolecularStandardizer import MolecularStandardizer
from typing import Any

# Customizable standardizer and custom parameters
# adapted from https://github.com/brsynth/RetroPathRL
simple_standardisation = {
    'REMOVE_ISOTOPE': False,
    'NEUTRALISE_CHARGE': False,
    'REMOVE_STEREO': True,
    'KEEP_BIGGEST': False,
    'ADD_HYDROGEN': True,
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


def custom_standardizer(mol,
                        REMOVE_ISOTOPE=True,
                        NEUTRALISE_CHARGE=True,
                        REMOVE_STEREO=False,
                        KEEP_BIGGEST=True,
                        ADD_HYDROGEN=True,
                        KEKULIZE=True,
                        NEUTRALISE_CHARGE_LATE=True):
    """Tunable sequence of filters for standardization.

    Operations will made in the following order:
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

    if KEKULIZE:
        mol = kekulize(mol)

    return mol


class CustomStandardizer(MolecularStandardizer):
    """...
    """
    simple_standardisation = {
        'REMOVE_ISOTOPE': False,
        'NEUTRALISE_CHARGE': False,
        'REMOVE_STEREO': True,
        'KEEP_BIGGEST': False,
        'ADD_HYDROGEN': True,
        'KEKULIZE': False,
        'NEUTRALISE_CHARGE_LATE': True}

    def __init__(self, params=None):
        """
        Parameters
        ----------
        params: dict, optional (default simple_standardization params)
            Parameters containing which steps of standardization to take.
        """

        super().__init__()
        if params is None:
            params = simple_standardisation
        self.params = params

    def _standardize(self, mol: Any):
        """Standardizes a molecule SMILES using a custom set of steps.
         Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
            RDKit Mol object
        Returns
        -------
        mol: str
            Standardized mol SMILES.
        """

        try:
            mol = custom_standardizer(mol, **self.params)
        except Exception as e:
            print(e)
            print('error in standardizing smile: ' + str(mol))

        return mol
