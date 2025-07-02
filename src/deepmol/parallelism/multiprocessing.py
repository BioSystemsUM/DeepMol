from abc import ABC, abstractmethod
import contextlib
from typing import Iterable

from joblib import Parallel, delayed
from tqdm import tqdm

from deepmol.loggers.logger import Logger

import joblib


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


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
        try:
            self._process_name = self.process.__self__.__class__.__name__
        except AttributeError:
            self._process_name = self.process.__name__

        self.logger = Logger()

    @property
    def process(self):
        """
        Returns the function to use for multiprocessing.
        """
        return self._process

    def run_iteratively(self, items: list):
        """
        Does not run multiprocessing due to an error pickling the process function or other.
        """
        if isinstance(items[0], tuple):

            for item in tqdm(items, desc=self._process_name):
                yield self.process(*item)
        else:
            for item in tqdm(items, desc=self._process_name):
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

        try:
            # verifying if the process is a zip and convert it to a list
            if isinstance(items, zip):
                items = list(items)

            # verifying if the first element is a tuple, if so one must use the args parameter *item
            if isinstance(items[0], tuple):
                parallel_callback = Parallel(backend="threading", n_jobs=self.n_jobs)
                with tqdm_joblib(tqdm(desc=self._process_name, total=len(items))):
                    results = parallel_callback(
                        delayed(self.process)(*item) for item in items)
            else:
                parallel_callback = Parallel(backend="threading", n_jobs=self.n_jobs)
                with tqdm_joblib(tqdm(desc=self._process_name, total=len(items))):
                    results = parallel_callback(
                        delayed(self.process)(item) for item in items)
                    
        except Exception as e:
            if "pickle" in str(e):
                self.logger.warning(f"Failed to pickle process {self.process.__name__} function. Processing the input "
                                    f"iteratively instead.")
                results = self.run_iteratively(items)
            else:
                raise e

        return results
