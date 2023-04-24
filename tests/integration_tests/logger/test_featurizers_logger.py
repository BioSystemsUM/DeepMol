import os

from deepmol.compound_featurization import MorganFingerprint
from deepmol.loggers.logger import Logger
from integration_tests.logger.test_logger import TestLogger

from tests import TEST_DIR


class TestLoggerFeaturizer(TestLogger):

    def test_featurizers_disable_enable_logging(self):
        Logger.disable()
        self.logger.set_file_path(os.path.join(TEST_DIR, "test2.log"))
        MorganFingerprint().featurize(self.big_dataset_to_test)
        self.assertTrue(not os.path.exists(os.path.join(TEST_DIR, "test2.log")))

        Logger().enable()
        MorganFingerprint().featurize(self.big_dataset_to_test)
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, "test2.log")))
        os.remove(os.path.join(TEST_DIR, "test2.log"))


