from deepmol.compound_featurization import MolecularFeaturizer
from rdkit.Chem import Mol
import numpy as np
from biosynfoni import Biosynfoni
from biosynfoni.subkeys import substructureSmarts

class BiosynfoniKeys(MolecularFeaturizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features_names = []
        for key in substructureSmarts.keys():
            self.features_names.append(substructureSmarts[key]["name"])        
        
    
    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Featurize a molecule using the Biosynfoni fingerprint.

        Args:
            mol (Mol): RDKit Mol object.
        Returns:
            np.ndarray: Fingerprint of the molecule.
        """

        fp = Biosynfoni(mol).fingerprint
        fp = np.array(fp)
        return fp