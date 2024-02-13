import os
from unittest import TestCase

from deepmol.loggers import Logger
from deepmol.parallelism.multiprocessing import JoblibMultiprocessing


def divide(a, b):
    return a / b


class TestMultiProcessing(TestCase):

    def tearDown(self) -> None:
        # Close logger file handlers to release the file
        singleton_instance = Logger()
        singleton_instance.close_handlers()
        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')

    def test_multiprocessing_to_fail(self):
        def f(x):
            def g():
                nonlocal x
                x += 1
                return x

            return g

        JoblibMultiprocessing(n_jobs=5, process=f).run([1, 2, 3])

    def test_multiprocessing_with_tuple_input(self):
        def f_(x):
            return x

        JoblibMultiprocessing(n_jobs=5, process=f_).run([(1,), (2,), (3,)])
        JoblibMultiprocessing(n_jobs=5, process=f_).run_iteratively([(1,), (2,), (3,)])

    def test_multiprocessing_with_zip_input(self):
        def f_(x):
            return x

        JoblibMultiprocessing(n_jobs=5, process=f_).run(zip([1, 2, 3]))

    def test_multiprocessing_to_fail_in_exception(self):

        with self.assertRaises(ZeroDivisionError):
            JoblibMultiprocessing(n_jobs=5, process=divide).run([(1, 0), (2, 1), (3, 2)])

        JoblibMultiprocessing(n_jobs=5, process=divide).run_iteratively([(1, 1), (2, 1), (3, 2)])
