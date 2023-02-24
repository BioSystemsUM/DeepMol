from rdkit.Chem import Mol

from deepmol.standardizer import MolecularStandardizer
from deepmol.standardizer._utils import basic_standardizer


class BasicStandardizer(MolecularStandardizer):
    """
    Standardizes a molecule SMILES using the SanitizeMol rdkit method.
    """

    def _standardize(self, mol: Mol) -> Mol:
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
        return basic_standardizer(mol)
