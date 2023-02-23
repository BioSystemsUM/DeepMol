import logging
import os
from unittest import TestCase

from deepmol.loggers.logger import Logger
from tests import TEST_DIR


class TestLogger(TestCase):

    def setUp(self) -> None:
        self.log_file_name = os.path.join(TEST_DIR, "test.log")
        self.logger = Logger(file_path=self.log_file_name + "2", level=logging.DEBUG)

    def delete_file(self) -> None:
        if os.path.exists(self.log_file_name + "2"):
            os.remove(self.log_file_name + "2")

    def test_logger(self):
        Logger(file_path=self.log_file_name).info("Test")
        self.assertTrue(os.path.exists(self.log_file_name + "2"))
        with open(self.log_file_name + "2", "r") as f:
            self.assertIn("Test", f.readline())

    def test_singleton(self):
        logger1 = Logger(file_path=self.log_file_name)
        logger2 = Logger(file_path=self.log_file_name)
        self.assertEqual(logger1, logger2)
        self.assertEqual(logger1, self.logger)
        self.assertEqual(logger2, self.logger)

    def test_logger_at_class_level(self):
        Logger(file_path=self.log_file_name + "2", level=logging.DEBUG).info("Test")

        class Test:
            logger = Logger(file_path=self.log_file_name + "2", level=logging.DEBUG)

            def method(self):
                self.logger.info("TEST inside class")

        Test().method()

        self.assertTrue(os.path.exists(self.log_file_name + "2"))
        with open(self.log_file_name + "2", "r") as f:
            f.readline()
            self.assertIn("Test", f.readline())
            self.assertIn("TEST inside class", f.readline())

        self.delete_file()

    def test_pickle(self):
        import pickle
        pickle.dumps(self.logger)
