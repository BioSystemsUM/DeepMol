import os
from typing import Iterable

import numpy as np
from rdkit.Chem import Mol

from compound_featurization.base_featurizer import MolecularFeaturizer


class MixedFeaturizer(MolecularFeaturizer):

    def __init__(self, featurizers: Iterable[MolecularFeaturizer]):

        super().__init__()
        self.featurizers = featurizers

    def _featurize(self, mol: Mol) -> np.ndarray:
        """Featurization with mix of featurizers
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of 3D Autocorrelation descriptors
        """

        try:
            final_features = np.array([])
            for featurizer in self.featurizers:
                current_features = featurizer._featurize(mol)
                final_features = np.concatenate((final_features, current_features))

        except Exception:
            print('error in smile: ' + str(mol))
            final_features = np.empty(80, dtype=float)
            final_features[:] = np.NaN

        final_features = np.asarray(final_features, dtype=np.float)

        return final_features
