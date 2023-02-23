import re
from abc import ABC, abstractmethod
from typing import Iterable

from joblib import Parallel, delayed

from deepmol.loggers.logger import Logger


class MultiprocessingClass(ABC):
    """
    Base class for multiprocessing.
    """

    def __init__(self, n_jobs: int = -1, process: callable = None):
        """
        Constructor for the MultiprocessingClass class.

        Parameters
        ----------
        n_jobs: int
            The number of jobs to use for multiprocessing. If -1, all available cores are used.
        process: callable
            The function to use for multiprocessing.
        """
        self.n_jobs = n_jobs
        self._process = process

        self.logger = Logger()

    @property
    def process(self):
        """
        Returns the function to use for multiprocessing.
        """
        return self._process

    def run_iteratively(self, items: list):
        """
        Does not run multiprocessing due to an error pickleling the process function or other.
        """
        if isinstance(items[0], tuple):

            for item in items:
                yield self.process(*item)
        else:
            for item in items:
                yield self.process(item)

    @abstractmethod
    def run(self, items: Iterable) -> Iterable:
        """
        Runs the multiprocessing.

        Parameters
        ----------
        items: Iterable
            The items to use for multiprocessing.

        Returns
        -------
        results: Iterable
            The results of the multiprocessing.
        """
        pass


class JoblibMultiprocessing(MultiprocessingClass):
    """
    Multiprocessing class using joblib.
    """

    def run(self, items: Iterable) -> Iterable:
        """
        Runs the multiprocessing.

        Parameters
        ----------
        items: Iterable
            The items to use for multiprocessing.

        Returns
        -------
        results: Iterable
            The results of the multiprocessing.
        """
        # TODO: Add support for progress bar

        try:
            # verifying if the process is a zip and convert it to a list
            if isinstance(items, zip):
                items = list(items)

            # verifying if the first element is a tuple, if so one must use the args parameter *item
            if isinstance(items[0], tuple):
                results = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(delayed(self.process)(*item)
                                                                                  for item in items)
            else:
                results = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(delayed(self.process)(item)
                                                                                  for item in items)

        except TypeError as e:
            if re.match("cannot pickle '.*' object", str(e)):
                self.logger.warning("Failed to pickle process function. Processing the input iteratively instead.")
                results = self.run_iteratively(items)
            else:
                raise e

        return results
