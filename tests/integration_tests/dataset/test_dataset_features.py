from deepmol.compound_featurization import MorganFingerprint, LayeredFingerprint, TwoDimensionDescriptors
from integration_tests.dataset.test_dataset import TestDataset


class TestDatasetFeaturizers(TestDataset):

    def test_clear_cached_property_after_generating_features(self):

        MorganFingerprint().featurize(self.small_dataset_to_test, inplace=True)

        self.assertEqual(self.small_dataset_to_test.X.shape, (13, 2048))

        LayeredFingerprint(fpSize=1024).featurize(self.small_dataset_to_test, inplace=True)

        self.assertEqual(self.small_dataset_to_test.X.shape, (13, 1024))

        TwoDimensionDescriptors().featurize(self.small_dataset_to_test, inplace=True)

