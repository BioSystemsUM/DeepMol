import os

from deepmol.standardizer import BasicStandardizer
from integration_tests.logger.test_logger import TestLogger

from tests import TEST_DIR


class TestLoggerStandardizers(TestLogger):

    def test_standardizers(self):
        BasicStandardizer(n_jobs=2).standardize(self.small_dataset_to_test)

        with open(os.path.join(TEST_DIR, "test.log"), 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            self.assertIn("Standardizer BasicStandardizer initialized", lines[0])
