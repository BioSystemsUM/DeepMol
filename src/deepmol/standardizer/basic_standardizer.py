from rdkit.Chem import SanitizeMol, Mol

from deepmol.standardizer import MolecularStandardizer


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


class BasicStandardizer(MolecularStandardizer):
    """
    Standardizes a molecule SMILES using the SanitizeMol rdkit method.
    """

    def _standardize(self, mol: Mol):
        """
        Standardizes a molecule SMILES using a custom set of steps.

        Parameters
        ----------
        mol: Mol
            RDKit Mol object

        Returns
        -------
        mol: str
            Standardized mol.
        """
        try:
            mol = basic_standardizer(mol)
        except Exception as e:
            print('error in standardizing smile: ' + str(mol))
        return mol
