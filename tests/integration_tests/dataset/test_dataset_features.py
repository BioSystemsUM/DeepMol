from deepmol.compound_featurization import MorganFingerprint, LayeredFingerprint, TwoDimensionDescriptors
from tests.integration_tests.dataset.test_dataset import TestDataset

import unittest

class TestDatasetFeaturizers(TestDataset):

    def test_clear_cached_property_after_generating_features(self):

        MorganFingerprint().featurize(self.small_dataset_to_test, inplace=True)

        self.assertEqual(self.small_dataset_to_test.X.shape, (13, 2048))

        LayeredFingerprint(fpSize=1024).featurize(self.small_dataset_to_test, inplace=True)

        self.assertEqual(self.small_dataset_to_test.X.shape, (13, 1024))

        TwoDimensionDescriptors().featurize(self.small_dataset_to_test, inplace=True)

    @unittest.skip("Requires too much memory")
    def test_dataset_with_similarity_matrix(self):
        from deepmol.compound_featurization import TanimotoSimilarityMatrix
        dataset_rows_number = len(self.small_dataset_to_test.mols)
        TanimotoSimilarityMatrix(n_molecules=4).featurize(self.small_dataset_to_test, inplace=True)
        self.assertEqual(dataset_rows_number, self.small_dataset_to_test._X.shape[0])
        self.assertEqual(4, self.small_dataset_to_test._X.shape[1])

