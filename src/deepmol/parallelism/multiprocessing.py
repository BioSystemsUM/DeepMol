from abc import ABC, abstractmethod
from typing import List, Iterable

from joblib import Parallel, delayed


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

    @property
    def process(self):
        """
        Returns the function to use for multiprocessing.
        """
        return self._process

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
        return results
