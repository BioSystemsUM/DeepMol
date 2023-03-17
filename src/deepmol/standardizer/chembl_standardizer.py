from chembl_structure_pipeline import standardizer
from rdkit.Chem import Mol

from deepmol.standardizer import MolecularStandardizer


class ChEMBLStandardizer(MolecularStandardizer):
    """
    Standardizes a molecule SMILES using the ChEMBL standardizer.
    https://github.com/chembl/ChEMBL_Structure_Pipeline
    """

    def _standardize(self, mol: Mol) -> Mol:
        """
        Standardizes a molecule SMILES using the ChEMBL standardizer.

        Parameters
        ----------
        mol: Mol
            RDKit Mol object

        Returns
        -------
        mol: Mol
            Standardized Mol.
        """
        mol = standardizer.standardize_mol(mol)
        mol, _ = standardizer.get_parent_mol(mol)
        return mol
