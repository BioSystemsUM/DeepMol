import os

from deepmol.compound_featurization import MorganFingerprint
from integration_tests.logger.test_logger import TestLogger

from tests import TEST_DIR


class TestLoggerFeaturizer(TestLogger):

    def test_featurizers_to_fail(self):
        MorganFingerprint().featurize(self.big_dataset_to_test)
        with open(os.path.join(TEST_DIR, "test.log"), 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 19)
            self.assertIn("COc1ccc(CCc2ccccc2CO)cc1O09y78", lines[0])

    def test_featurizers_one_job(self):
        MorganFingerprint(n_jobs=1).featurize(self.big_dataset_to_test)
        with open(os.path.join(TEST_DIR, "test.log"), 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 19)
            self.assertIn("COc1ccc(CCc2ccccc2CO)cc1O09y78", lines[0])