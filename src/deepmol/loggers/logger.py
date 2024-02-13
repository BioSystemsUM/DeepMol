import logging

import sys
from logging.handlers import TimedRotatingFileHandler

disabled_logger = False


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

        self.file_path = file_path
        self.level = level
        self.formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

        self.file_path_changed = False
        self.logger = None
        self.console_handler = None
        self.file_handler = None

        if not disabled_logger:
            self.create_handlers()

    @staticmethod
    def disable():
        """
        Disables the logger.
        """
        # disable all levels of logging
        logging.disable(logging.DEBUG)
        logging.disable(logging.INFO)
        logging.disable(logging.CRITICAL)
        logging.disable(logging.ERROR)
        logging.disable(logging.WARNING)

        global disabled_logger
        disabled_logger = True

    def enable(self):
        """
        Enables the logger.
        """
        # logging.disable(logging.NOTSET)
        global disabled_logger
        disabled_logger = False

        if self.file_path_changed:
            self.logger.removeHandler(self.file_handler)
            self.file_path_changed = False
            file_handler = TimedRotatingFileHandler(self.file_path, when='midnight')
            self.file_handler = file_handler
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)
            self.logger.setLevel(self.level)
        self.create_handlers()

    def create_handlers(self):
        """
        Creates the handlers for the logger.
        """

        if not self.logger:
            self.logger = logging.getLogger(self.file_path)
            self.console_handler = logging.StreamHandler(sys.stdout)
            self.console_handler.setFormatter(self.formatter)
            self.file_handler = TimedRotatingFileHandler(self.file_path, when='midnight')
            self.file_handler.setFormatter(self.formatter)
            if not self.logger.hasHandlers():
                self.logger.addHandler(self.file_handler)
                self.logger.addHandler(self.console_handler)
            self.logger.setLevel(self.level)

    def set_file_path(self, file_path: str):
        """
        Sets the file path of the logger.

        Parameters
        ----------
        file_path: str
            The path to the log file.
        """
        self.file_path = file_path
        self.file_path_changed = True
        if not disabled_logger:
            self.logger = logging.getLogger(self.file_path)
            self.logger.removeHandler(self.file_handler)

            file_handler = TimedRotatingFileHandler(file_path, when='midnight')
            self.file_handler = file_handler
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)
            self.logger.addHandler(self.console_handler)
            self.logger.setLevel(self.level)

    def set_level(self, level: int):
        """
        Sets the level of the logger.

        Parameters
        ----------
        level: int
            The level of the logger.
        """
        self.logger.setLevel(level)

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

    def close_handlers(self):
        """
        Closes the handlers of the logger.
        """
        if self.console_handler:
            self.console_handler.close()
        if self.file_handler:
            self.file_handler.close()

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
        if 'console_handler' in d:
            d['console_handler'] = 'console_handler'
        if 'file_handler' in d:
            d['file_handler'] = 'file_handler'

        return d

    def __setstate__(self, d: dict):
        """
        Sets the state of the logger.

        Parameters
        ----------
        d: dict
            The dictionary to set the state of the logger.

        """

        d['logger'] = logging.getLogger(d['file_path'])
        self.__dict__.update(d)
