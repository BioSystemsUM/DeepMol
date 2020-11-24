import tempfile
import numpy as np

from typing import Tuple, List, Optional, Iterator

from Dataset.Dataset import Dataset, NumpyDataset, CSVLoader

from sklearn.model_selection import KFold


class Splitter(object):
    """Splitters split up Datasets into pieces for training/validation/testing.
    In machine learning applications, it's often necessary to split up a dataset
    into training/validation/test sets. Or to k-fold split a dataset (that is,
    divide into k equal subsets) for cross-validation. The `Splitter` class is
    an abstract superclass for all splitters that captures the common API across
    splitter classes.
    Note that `Splitter` is an abstract superclass. You won't want to
    instantiate this class directly. Rather you will want to use a concrete
    subclass for your application.
    """

    #TODO: Possible upgrade: add directories input to save splits to file (update code)
    def k_fold_split(self,
                     dataset: Dataset,
                     k: int,
                     **kwargs) -> List[Tuple[Dataset, Dataset]]:
        """
        Parameters
        ----------
        dataset: Dataset
          Dataset to do a k-fold split
        k: int
          Number of folds to split `dataset` into.
        directories: List[str], optional (default None)
          List of length 2*k filepaths to save the result disk-datasets.
        Returns
        -------
        List[Tuple[Dataset, Dataset]]
          List of length k tuples of (train, cv) where `train` and `cv` are both `Dataset`.
        """
        print("Computing K-fold split")

        if isinstance(dataset, NumpyDataset) or isinstance(dataset, CSVLoader):
            ds = dataset
        else:
            ds = NumpyDataset.from_numpy(dataset.X, dataset.y, dataset.features, dataset.ids)

        kf = KFold(n_splits=k)

        train_datasets = []
        test_datasets = []
        for train_index, test_index in kf.split(ds.X):
            train_datasets.append(ds.select(train_index))
            test_datasets.append(ds.select(test_index))

        return list(zip(train_datasets, test_datasets))



    def train_valid_test_split(self,
                               dataset: Dataset,
                               frac_train: float = 0.8,
                               frac_valid: float = 0.1,
                               frac_test: float = 0.1,
                               seed: Optional[int] = None,
                               log_every_n: int = 1000,
                               **kwargs) -> Tuple[Dataset, Dataset, Dataset]:
        """ Splits self into train/validation/test sets.
        Returns Dataset objects for train, valid, test.
        Parameters
        ----------
        dataset: Dataset
          Dataset to be split.
        frac_train: float, optional (default 0.8)
          The fraction of data to be used for the training split.
        frac_valid: float, optional (default 0.1)
          The fraction of data to be used for the validation split.
        frac_test: float, optional (default 0.1)
          The fraction of data to be used for the test split.
        seed: int, optional (default None)
          Random seed to use.
        log_every_n: int, optional (default 1000)
          Controls the logger by dictating how often logger outputs
          will be produced.
        Returns
        -------
        Tuple[Dataset, Optional[Dataset], Dataset]
          A tuple of train, valid and test datasets as dc.data.Dataset objects.
        """

        print("Computing train/valid/test indices")
        train_inds, valid_inds, test_inds = self.split(
            dataset,
            frac_train=frac_train,
            frac_test=frac_test,
            frac_valid=frac_valid,
            seed=seed,
            log_every_n=log_every_n)

        train_dataset = dataset.select(train_inds)
        valid_dataset = dataset.select(valid_inds)
        test_dataset = dataset.select(test_inds)
        if isinstance(train_dataset, Dataset):
            train_dataset.memory_cache_size = 40 * (1 << 20)  # 40 MB

        return train_dataset, valid_dataset, test_dataset

    def train_test_split(self,
                        dataset: Dataset,
                        frac_train: float = 0.8,
                        seed: Optional[int] = None,
                        **kwargs) -> Tuple[Dataset, Dataset]:
        """Splits self into train/test sets.
        Returns Dataset objects for train/test.
        Parameters
        ----------
        dataset: data like object
          Dataset to be split.
        frac_train: float, optional (default 0.8)
          The fraction of data to be used for the training split.
        seed: int, optional (default None)
          Random seed to use.
        Returns
        -------
        Tuple[Dataset, Dataset]
          A tuple of train and test datasets as dc.data.Dataset objects.
        """

        train_dataset, _, test_dataset = self.train_valid_test_split(
            dataset,
            frac_train=frac_train,
            frac_test=1 - frac_train,
            frac_valid=0.,
            seed=seed,
            **kwargs)
        return train_dataset, test_dataset

    def split(self,
              dataset: Dataset,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              seed: Optional[int] = None,
              log_every_n: Optional[int] = None) -> Tuple:

        """Return indices for specified split
        Parameters
        ----------
        dataset: dc.data.Dataset
          Dataset to be split.
        seed: int, optional (default None)
          Random seed to use.
        frac_train: float, optional (default 0.8)
          The fraction of data to be used for the training split.
        frac_valid: float, optional (default 0.1)
          The fraction of data to be used for the validation split.
        frac_test: float, optional (default 0.1)
          The fraction of data to be used for the test split.
        log_every_n: int, optional (default None)
          Controls the logger by dictating how often logger outputs
          will be produced.
        Returns
        -------
        Tuple
          A tuple `(train_inds, valid_inds, test_inds)` of the indices (integers) for
          the various splits.
        """
        raise NotImplementedError


class RandomSplitter(Splitter):
    """Class for doing random data splits."""

    def split(self,
              dataset: Dataset,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              seed: Optional[int] = None,
              log_every_n: Optional[int] = None
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits internal compounds randomly into train/validation/test.
        Parameters
        ----------
        dataset: Dataset
          Dataset to be split.
        seed: int, optional (default None)
          Random seed to use.
        frac_train: float, optional (default 0.8)
          The fraction of data to be used for the training split.
        frac_valid: float, optional (default 0.1)
          The fraction of data to be used for the validation split.
        frac_test: float, optional (default 0.1)
          The fraction of data to be used for the test split.
        seed: int, optional (default None)
          Random seed to use.
        log_every_n: int, optional (default None)
          Log every n examples (not currently used).
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
          A tuple of train indices, valid indices, and test indices.
          Each indices is a numpy array.
        """

        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
        if seed is not None:
            np.random.seed(seed)
        num_datapoints = len(dataset)
        train_cutoff = int(frac_train * num_datapoints)
        valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
        shuffled = np.random.permutation(range(num_datapoints))
        return (shuffled[:train_cutoff], shuffled[train_cutoff:valid_cutoff], shuffled[valid_cutoff:])


