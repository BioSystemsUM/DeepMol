from copy import deepcopy

from deepmol.loggers import Logger
from integration_tests.logger.test_logger import TestLogger


class TestLoggerFeaturizer(TestLogger):
    def test_logger_with_deep_copy(self):
        d1 = deepcopy(self.big_dataset_to_test)

        self.assertIsInstance(d1.logger, Logger)
