import tempfile
import numpy as np

from typing import Tuple, List, Optional, Iterator, Type

from Datasets.Datasets import Dataset, NumpyDataset

from sklearn.model_selection import KFold, StratifiedKFold

from collections import Counter


class Splitter(object):
    """Splitters split up Datasets into pieces for training/validation/testing.
    In machine learning applications, it's often necessary to split up a dataset
    into training/validation/test sets. Or to k-fold split a dataset for cross-validation.
    """

    #TODO: Possible upgrade: add directories input to save splits to file (update code)
    def k_fold_split(self,
                     dataset: Dataset,
                     k: int,
                     seed: Optional[int] = None, # added
                     **kwargs) -> List[Tuple[Dataset, Dataset]]:
        """
        Parameters
        ----------
        dataset: Dataset
          Dataset to do a k-fold split
        k: int
          Number of folds to split `dataset` into.
        Returns
        -------
        List[Tuple[Dataset, Dataset]]
          List of length k tuples of (train, test) where `train` and `test` are both `Dataset`.
        """
        print("Computing K-fold split")

        if isinstance(dataset, NumpyDataset):
            ds = dataset
        else:

            ds = NumpyDataset(dataset.mols, dataset.X, dataset.y, dataset.ids, dataset.features2keep, dataset.n_tasks)

        # kf = KFold(n_splits=k)
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)

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
        """ Splits a Dataset into train/validation/test sets.
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
        Tuple[Dataset, Dataset, Dataset]
          A tuple of train, valid and test datasets as Dataset objects.
        """

        #print("Computing train/valid/test indices")
        train_inds, valid_inds, test_inds = self.split(dataset,
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
          A tuple of train and test datasets as Dataset objects.
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
        log_every_n: int, optional (default None)
          Controls the logger by dictating how often logger outputs
          will be produced.
        Returns
        -------
        Tuple
          A tuple `(train_inds, valid_inds, test_inds)` of the indices for
          the various splits.
        """
        raise NotImplementedError


class RandomSplitter(Splitter):
    """Class for doing random data splits."""

    def split(self,
              dataset: Type[Dataset],
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              seed: Optional[int] = None,
              log_every_n: Optional[int] = None
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits randomly into train/validation/test.
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
        #num_datapoints = len(dataset)
        num_datapoints = dataset.len_mols()
        train_cutoff = int(frac_train * num_datapoints)
        valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
        shuffled = np.random.permutation(range(num_datapoints))
        return (shuffled[:train_cutoff], shuffled[train_cutoff:valid_cutoff], shuffled[valid_cutoff:])


class SingletaskStratifiedSplitter(Splitter):
    # TODO: comments
    """Class for doing data splits by stratification on a single task.
    Examples
    --------
    n_samples = 100
    n_features = 10
    n_tasks = 10
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples, n_tasks)
    w = np.ones_like(y)
    dataset = DiskDataset.from_numpy(np.ones((100,n_tasks)), np.ones((100,n_tasks)))
    splitter = SingletaskStratifiedSplitter(task_number=5)
    train_dataset, test_dataset = splitter.train_test_split(dataset)
    """

    def k_fold_split(self,
                     dataset: Dataset,
                     k: int,
                     seed: Optional[int] = None,
                     log_every_n: Optional[int] = None,
                     **kwargs) -> List[Dataset]:
        # TODO: comments
        """
        Splits compounds into k-folds using stratified sampling.
        Overriding base class k_fold_split.
        Parameters
        ----------
        dataset: Dataset
          Dataset to be split.
        k: int
          Number of folds to split `dataset` into.
        seed: int, optional (default None)
          Random seed to use.
        log_every_n: int, optional (default None)
          Log every n examples (not currently used).
        Returns
        -------
        fold_datasets: List[Dataset]
          List of dc.data.Dataset objects
        """

        print("Computing Stratified K-fold split")

        if isinstance(dataset, NumpyDataset):
            ds = dataset
        else:
            ds = NumpyDataset(dataset.mols, dataset.X, dataset.y, dataset.ids, dataset.features2keep, dataset.n_tasks)

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed) # changed so that users can define the seed

        train_datasets = []
        test_datasets = []
        for train_index, test_index in skf.split(ds.X, ds.y):
            train_datasets.append(ds.select(train_index))
            test_datasets.append(ds.select(test_index))

        return list(zip(train_datasets, test_datasets))


    def split(self,
              dataset: Dataset,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              seed: Optional[int] = None,
              log_every_n: Optional[int] = None
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits compounds into train/validation/test using stratified sampling.
        Parameters
        ----------
        dataset: Dataset
          Dataset to be split.
        frac_train: float, optional (default 0.8)
          Fraction of dataset put into training data.
        frac_valid: float, optional (default 0.1)
          Fraction of dataset put into validation data.
        frac_test: float, optional (default 0.1)
          Fraction of dataset put into test data.
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

        np.testing.assert_equal(frac_train + frac_valid + frac_test, 1.)
        np.testing.assert_equal(10 * frac_train + 10 * frac_valid + 10 * frac_test, 10.)

        if seed is not None:
            np.random.seed(seed)

        sortidx = np.argsort(dataset.y)

        split_cd = 10
        train_cutoff = int(np.round(frac_train * split_cd))
        valid_cutoff = int(np.round(frac_valid * split_cd)) + train_cutoff

        train_idx = np.array([])
        valid_idx = np.array([])
        test_idx = np.array([])

        while sortidx.shape[0] >= split_cd:
            sortidx_split, sortidx = np.split(sortidx, [split_cd])
            shuffled = np.random.permutation(range(split_cd))
            train_idx = np.hstack([train_idx, sortidx_split[shuffled[:train_cutoff]]])
            valid_idx = np.hstack([valid_idx, sortidx_split[shuffled[train_cutoff:valid_cutoff]]])
            test_idx = np.hstack([test_idx, sortidx_split[shuffled[valid_cutoff:]]])

        # Append remaining examples to train
        if sortidx.shape[0] > 0:
            np.hstack([train_idx, sortidx])

        return (list(map(int, train_idx)), list(map(int, valid_idx)), list(map(int, test_idx)))


