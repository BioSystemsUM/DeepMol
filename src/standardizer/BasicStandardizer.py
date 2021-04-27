from standardizer.MolecularStandardizer import MolecularStandardizer
from rdkit.Chem import SanitizeMol
from typing import Any

# Performs only Sanitization
# (Kekulize, check valencies, set aromaticity, conjugation and hybridization)
def basic_standardizer(mol):
    SanitizeMol(mol)
    return mol

class BasicStandardizer(MolecularStandardizer):
    """...
    """

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
            mol = basic_standardizer(mol)
        except Exception as e:
            print('error in standardizing smile: ' + str(mol))
        return mol