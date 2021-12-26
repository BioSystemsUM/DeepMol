from copy import copy
from unittest import TestCase

from compoundFeaturization.mixedDescriptors import MixedFeaturizer
from compoundFeaturization.mol2vec import Mol2Vec
from compoundFeaturization.rdkitDescriptors import All3DDescriptors
from compoundFeaturization.rdkitFingerprints import MorganFingerprint, AtomPairFingerprintCallbackHash
from scalers.sklearnScalers import StandardScaler
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase

import numpy as np


class TestMixedFeaturizer(FeaturizerTestCase, TestCase):
    def test_featurize(self):
        dataset_rows_number = len(self.mini_dataset_to_test.mols)
        descriptors = [All3DDescriptors(generate_conformers=True), MorganFingerprint()]
        MixedFeaturizer(featurizers=descriptors).featurize(self.mini_dataset_to_test)
        self.assertEqual(dataset_rows_number, self.mini_dataset_to_test.X.shape[0])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mini_dataset_to_test.mols)
        to_add = np.zeros(4)

        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))

        dataset = copy(self.mini_dataset_to_test)
        descriptors = [All3DDescriptors(generate_conformers=True), MorganFingerprint()]
        MixedFeaturizer(featurizers=descriptors).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

    def test_mixed_featurizer(self):
        atom_pair = AtomPairFingerprintCallbackHash(nBits=1024, includeChirality=True)
        scaler = StandardScaler()
        moltovec = Mol2Vec()
        featurize_method = MixedFeaturizer(featurizers=[atom_pair, moltovec])
        featurize_method.featurize(self.mini_dataset_to_test)
        columns = [i for i in range(1024, 1324)]
        scaler.fit_transform(self.mini_dataset_to_test, columns)
        scaler.transform(self.mini_dataset_to_test, columns)

