from copy import copy
from unittest import TestCase

from deepmol.compound_featurization import MixedFeaturizer
from deepmol.compound_featurization.mol2vec import Mol2Vec
from deepmol.compound_featurization.rdkit_descriptors import All3DDescriptors
from deepmol.compound_featurization import MorganFingerprint, \
    AtomPairFingerprint
from deepmol.scalers import StandardScaler
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase

import numpy as np


class TestMixedFeaturizer(FeaturizerTestCase, TestCase):
    def test_featurize(self):
        dataset_rows_number = len(self.mini_dataset_to_test.mols)
        descriptors = [All3DDescriptors(mandatory_generation_of_conformers=True), MorganFingerprint()]
        MixedFeaturizer(featurizers=descriptors).featurize(self.mini_dataset_to_test)
        self.assertEqual(dataset_rows_number, self.mini_dataset_to_test.X.shape[0])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mini_dataset_to_test.mols)
        to_add = np.zeros(4)
        ids_to_add = np.array([5, 6, 7, 8])

        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))
        self.mini_dataset_to_test.ids = np.concatenate((self.mini_dataset_to_test.y, ids_to_add))

        dataset = copy(self.mini_dataset_to_test)
        descriptors = [All3DDescriptors(mandatory_generation_of_conformers=True), MorganFingerprint()]
        MixedFeaturizer(featurizers=descriptors).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

    def test_mixed_featurizer(self):
        atom_pair = AtomPairFingerprint(nBits=1024, includeChirality=True)
        scaler = StandardScaler()
        moltovec = Mol2Vec()
        featurize_method = MixedFeaturizer(featurizers=[moltovec, atom_pair])
        featurize_method.featurize(self.mini_dataset_to_test)
        columns = [i for i in range(300)]

        scaler.fit_transform(self.mini_dataset_to_test, columns)
        scaler.transform(self.mini_dataset_to_test, columns)

        self.assertEqual(self.mini_dataset_to_test.X.shape[1], 1324)
