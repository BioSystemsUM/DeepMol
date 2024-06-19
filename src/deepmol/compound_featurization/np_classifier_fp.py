

from deepmol.compound_featurization.base_featurizer import MolecularFeaturizer
from rdkit.Chem import rdMolDescriptors, Mol

import numpy as np

class NPClassifierFP(MolecularFeaturizer):

    def __init__(self, radius: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.radius = radius
        self.feature_names = [f'npclassifier_{i}' for i in range(2048*(self.radius+1))]

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Calculate morgan fingerprint for a single molecule.

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
          A numpy array of circular fingerprint.
        """
        binary = np.zeros((2048*(self.radius)), int)
        formula = np.zeros((2048),int)
        
        mol_bi = {}
        for r in range(self.radius+1):
            mol_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=r, bitInfo=mol_bi, nBits = 2048)
            mol_bi_QC = []
            for i in mol_fp.GetOnBits():
                num_ = len(mol_bi[i])
                for j in range(num_):
                    if mol_bi[i][j][1] == r:
                        mol_bi_QC.append(i)
                        break

            if r == 0:
                for i in mol_bi_QC:
                    formula[i] = len([k for k in mol_bi[i] if k[1]==0])
            else:
                for i in mol_bi_QC:
                    binary[(2048*(r-1))+i] = len([k for k in mol_bi[i] if k[1]==r])

        return np.concatenate((binary, formula), dtype=np.float32)
        

        

