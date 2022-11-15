from rdkit.Chem import Mol, Cleanup, SanitizeMol, SanitizeFlags, AssignStereochemistry

from deepmol.standardizer import MolecularStandardizer
from deepmol.standardizer.utils_standardization import remove_isotope_info, uncharge, remove_stereo, keep_biggest, \
    add_hydrogens, kekulize

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


def custom_standardizer(mol: Mol,
                        REMOVE_ISOTOPE: bool = True,
                        NEUTRALISE_CHARGE: bool = True,
                        REMOVE_STEREO: bool = False,
                        KEEP_BIGGEST: bool = True,
                        ADD_HYDROGEN: bool = True,
                        KEKULIZE: bool = True,
                        NEUTRALISE_CHARGE_LATE: bool = True):
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

    if KEKULIZE:
        mol = kekulize(mol)

    return mol


class CustomStandardizer(MolecularStandardizer):
    """
    Standardizes a molecule using a custom set of steps.
    """
    simple_standardisation = {
        'REMOVE_ISOTOPE': False,
        'NEUTRALISE_CHARGE': False,
        'REMOVE_STEREO': True,
        'KEEP_BIGGEST': False,
        'ADD_HYDROGEN': True,
        'KEKULIZE': False,
        'NEUTRALISE_CHARGE_LATE': True}

    def __init__(self, params: dict = None):
        """
        Initializes the standardizer.

        Parameters
        ----------
        params: dict
            Dictionary containing which steps of standardization to make.
        """
        super().__init__()
        if params is None:
            params = simple_standardisation
        self.params = params

    def _standardize(self, mol: Mol) -> Mol:
        """
        Standardizes a molecule SMILES using a custom set of steps.

        Parameters
        ----------
        mol: Mol
            RDKit Mol object

        Returns
        -------
        mol: Mol
            Standardized Mol.
        """
        try:
            mol = custom_standardizer(mol, **self.params)
        except Exception as e:
            print(e)
            print('error in standardizing smile: ' + str(mol))
        return mol
