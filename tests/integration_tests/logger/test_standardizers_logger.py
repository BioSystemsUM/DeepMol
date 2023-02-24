import os

from deepmol.standardizer import BasicStandardizer
from integration_tests.logger.test_logger import TestLogger

from tests import TEST_DIR


class TestLoggerStandardizers(TestLogger):

    def test_standardizers(self):
        self.logger.set_file_path(os.path.join(TEST_DIR, "test_standardizers.log"))
        BasicStandardizer(n_jobs=2).standardize(self.small_dataset_to_test)

        with open(os.path.join(TEST_DIR, "test_standardizers.log"), 'r') as f:
            lines = f.readlines()
            self.assertIn("Standardizer BasicStandardizer initialized", lines[0])
