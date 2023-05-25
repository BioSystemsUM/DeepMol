from typing import Iterable

import numpy as np
from rdkit.Chem import Mol

from deepmol.compound_featurization import MolecularFeaturizer


class MixedFeaturizer(MolecularFeaturizer):
    """
    Class to perform multiple types of featurizers.
    Features from different featurizers are concatenated.
    """

    def __init__(self, featurizers: Iterable[MolecularFeaturizer], **kwargs) -> None:
        """
        Parameters
        ----------
        featurizers: Iterable[MolecularFeaturizer]
            Iterable of featurizer to use to create features.
        """
        super().__init__(**kwargs)
        self.featurizers = featurizers
        self.feature_names = [name for featurizer in self.featurizers for name in featurizer.feature_names]

    def _featurize(self, mol: Mol):
        """
        Featurization with mix of featurizers.

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        np.ndarray
          A numpy array of concatenated features.
        """
        final_features = np.array([])
        for featurizer in self.featurizers:
            current_features = featurizer._featurize(mol)
            final_features = np.concatenate((final_features, current_features))

        final_features = np.asarray(final_features, dtype=np.float32)
        return final_features
