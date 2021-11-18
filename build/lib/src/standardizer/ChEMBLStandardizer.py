from chembl_structure_pipeline import standardizer
from standardizer.MolecularStandardizer import MolecularStandardizer
from typing import Any


class ChEMBLStandardizer(MolecularStandardizer):
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
            mol = standardizer.standardize_mol(mol)
            mol, _ = standardizer.get_parent_mol(mol)
        except Exception as e:
            print('error in standardizing smile: ' + str(mol))

        return mol
