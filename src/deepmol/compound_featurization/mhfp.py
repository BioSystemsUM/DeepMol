from rdkit.Chem import Mol

from deepmol.compound_featurization import MolecularFeaturizer
from deepmol.compound_featurization._mhfp import MHFPEncoder


class MHFP(MolecularFeaturizer):
    """
    MHFP featurizer class. This module contains the MHFP encoder, which is used to encode SMILES and RDKit
    molecule instances as MHFP fingerprints.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder = MHFPEncoder()
        self.feature_names = [f'mhfp_{i}' for i in range(2048)]

    def _featurize(self, mol: Mol):
        return self.encoder.encode_mol(mol)
