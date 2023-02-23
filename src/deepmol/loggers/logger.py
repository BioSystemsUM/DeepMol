import logging

import sys
from logging.handlers import TimedRotatingFileHandler


class SingletonMeta(type):
    """
    Singleton metaclass. The singleton lets you ensure that a class has only one instance,
    while providing a global access point to this instance.
    """

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
        """
        Uses the logger from the logging module.

        Parameters
        ----------
        file_path: str
            The path to the log file.
        level: int
            The level of the logger.
        """
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
        """
        Log a message with severity 'INFO' on the root logger.

        Parameters
        ----------
        msg: str
            The message to log.
        kwargs: dict
            The keyword arguments to pass to the logger.

        Returns
        -------

        """
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        """
        Log a message with severity 'WARNING' on the root logger.

        Parameters
        ----------
        msg: str
            The message to log.
        kwargs: dict
            The keyword arguments to pass to the logger.
        """
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs):
        """
        Log a message with severity 'ERROR' on the root logger.

        Parameters
        ----------
        msg: str
            The message to log.
        kwargs: dict
            The keyword arguments to pass to the logger.
        """
        self.logger.error(msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        """
        Log a message with severity 'CRITICAL' on the root logger.

        Parameters
        ----------
        msg: str
            The message to log.
        kwargs: dict
            The keyword arguments to pass to the logger.
        """
        self.logger.critical(msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        """
        Log a message with severity 'DEBUG' on the root logger.

        Parameters
        ----------
        msg: str
            The message to log.
        kwargs: dict
            The keyword arguments to pass to the logger.
        """
        self.logger.debug(msg, **kwargs)

    def __getstate__(self) -> dict:
        """
        Returns the state of the logger.
        It replaces the logger, console_handler and file_handler with their names to be picklezable, otherwise,
        it would not be possible to pickle the logger for multiprocessing.

        Returns
        -------
        d: dict
            The state of the logger.
        """
        d = self.__dict__.copy()
        d['logger'] = d['logger'].name
        d['console_handler'] = d['console_handler'].name
        d['file_handler'] = d['file_handler'].name

        return d

    def __setstate__(self, d: dict):
        """
        Sets the state of the logger.

        Parameters
        ----------
        d: dict
            The dictionary to set the state of the logger.

        """
        self.__dict__.update(d)
