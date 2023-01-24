from abc import ABC, abstractmethod
from joblib import Parallel, delayed


class MultiprocessingClass(ABC):

    def __init__(self, n_jobs: int = -1, process: callable = None):
        self.n_jobs = n_jobs
        self._process = process

    @property
    def process(self):
        return self._process

    @abstractmethod
    def run(self, items):
        pass


class JoblibMultiprocessing(MultiprocessingClass):

    def run(self, items: list):
        # Implement the processing logic for a single item here
        results = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(delayed(self.process)(item) for item in items)
        return results
