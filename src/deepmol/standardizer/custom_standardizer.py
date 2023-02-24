from rdkit.Chem import Mol

from deepmol.standardizer import MolecularStandardizer
from deepmol.standardizer._utils import simple_standardisation, custom_standardizer


class CustomStandardizer(MolecularStandardizer):
    """
    Standardizes a molecule using a custom set of steps.
    """

    def __init__(self, params: dict = None, **kwargs) -> None:
        """
        Initializes the standardizer.

        Parameters
        ----------
        params: dict
            Dictionary containing which steps of standardization to make.
        kwargs:
            Keyword arguments for the parent class.
        """
        super().__init__(**kwargs)
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
        return custom_standardizer(mol, **self.params)
