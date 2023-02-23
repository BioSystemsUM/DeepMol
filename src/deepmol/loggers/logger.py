import logging

import sys
from logging.handlers import TimedRotatingFileHandler


class SingletonMeta(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Logger(metaclass=SingletonMeta):

    def __init__(self, file_path: str = "deepmol.log", level: int = logging.DEBUG) -> None:
        self.logger = logging.getLogger(file_path)

        formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setFormatter(formatter)
        self.file_handler = TimedRotatingFileHandler(file_path, when='midnight')
        self.file_handler.setFormatter(formatter)

        self.logger.setLevel(level)
        self.logger.addHandler(self.file_handler)
        self.logger.propagate = False

    def info(self, msg: str, **kwargs):
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self.logger.error(msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        self.logger.critical(msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, **kwargs)

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
