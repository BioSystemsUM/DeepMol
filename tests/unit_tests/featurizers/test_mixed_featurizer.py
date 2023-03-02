from copy import copy
from unittest import TestCase

from deepmol.compound_featurization import MixedFeaturizer
from deepmol.compound_featurization.mol2vec import Mol2Vec
from deepmol.compound_featurization.rdkit_descriptors import All3DDescriptors
from deepmol.compound_featurization import MorganFingerprint, \
    AtomPairFingerprint
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase


class TestMixedFeaturizer(FeaturizerTestCase, TestCase):
    def test_featurize(self):
        dataset_rows_number = len(self.mock_dataset.mols)
        descriptors = [All3DDescriptors(mandatory_generation_of_conformers=True), MorganFingerprint()]
        MixedFeaturizer(featurizers=descriptors).featurize(self.mock_dataset)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mock_dataset_with_invalid.mols) - 1  # one mol has invalid smiles
        dataset = copy(self.mock_dataset_with_invalid)
        descriptors = [All3DDescriptors(mandatory_generation_of_conformers=True), MorganFingerprint()]
        MixedFeaturizer(featurizers=descriptors).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

    def test_mixed_featurizer(self):
        atom_pair = AtomPairFingerprint(nBits=1024, includeChirality=True)
        moltovec = Mol2Vec()
        featurize_method = MixedFeaturizer(featurizers=[moltovec, atom_pair])
        featurize_method.featurize(self.mock_dataset)
        columns = [i for i in range(300)]

        self.mock_scaler.fit_transform(self.mock_dataset, columns)
        self.mock_scaler.transform(self.mock_dataset, columns)

        self.assertEqual(self.mock_dataset._X.shape[1], 1324)
