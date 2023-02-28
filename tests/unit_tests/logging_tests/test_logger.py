import logging
import os
from unittest import TestCase

from deepmol.loggers.logger import Logger
from tests import TEST_DIR


class TestLogger(TestCase):

    def setUp(self) -> None:
        self.log_file_name = os.path.join(TEST_DIR, "test3.log")
        self.logger = Logger(file_path=self.log_file_name, level=logging.DEBUG)
        self.logger.set_file_path(self.log_file_name)

    def delete_file(self) -> None:
        if os.path.exists(self.log_file_name):
            os.remove(self.log_file_name)

    def test_logger(self):
        Logger(file_path=self.log_file_name).info("Test")
        self.assertTrue(os.path.exists(self.log_file_name))
        with open(self.log_file_name, "r") as f:
            self.assertIn("Test", f.readline())

    def test_logger_set_level(self):

        self.logger = Logger()
        test4 = os.path.join(TEST_DIR, "test4.log")
        self.logger.set_file_path(test4)

        self.logger.set_level(logging.INFO)
        self.assertEqual(self.logger.logger.level, logging.INFO)

    def test_logger_set_file_path(self):
        self.logger.set_file_path(self.log_file_name + "2")
        self.assertEqual(self.logger.logger.name, self.log_file_name + "2")
        self.logger.info("Test")
        self.assertTrue(os.path.exists(self.log_file_name + "2"))
        os.remove(self.log_file_name + "2")

    def test_singleton(self):
        logger1 = Logger(file_path=self.log_file_name)
        logger2 = Logger(file_path=self.log_file_name)
        self.assertEqual(logger1, logger2)
        self.assertEqual(logger1, self.logger)
        self.assertEqual(logger2, self.logger)

    def test_pickling(self):
        import pickle
        pickle.dumps(self.logger)

    def test_warning(self):
        self.logger.warning("Test")
        self.assertTrue(os.path.exists(self.log_file_name))
        with open(self.log_file_name, "r") as f:
            self.assertIn("Test", f.readline())

    def test_error(self):
        self.logger.error("Test")
        self.assertTrue(os.path.exists(self.log_file_name))
        with open(self.log_file_name, "r") as f:
            self.assertIn("Test", f.readline())

    def test_debug(self):
        self.logger.debug("Test")
        self.assertTrue(os.path.exists(self.log_file_name))
        with open(self.log_file_name, "r") as f:
            self.assertIn("Test", f.readline())

    def test_critical(self):
        self.logger.critical("Test")
        self.assertTrue(os.path.exists(self.log_file_name))
        with open(self.log_file_name, "r") as f:
            self.assertIn("Test", f.readline())

    @classmethod
    def tearDownClass(cls):
        log_file_name = os.path.join(TEST_DIR, "test3.log")
        if os.path.exists(log_file_name):
            os.remove(log_file_name)
