from typing import List, Tuple

import numpy as np

from deepmol.datasets import Dataset
from deepmol.splitters.splitters import Splitter

try:
    from skmultilearn.model_selection import IterativeStratification
except ImportError:
    pass


class MultiTaskStratifiedSplitter(Splitter):

    def split(self, dataset: Dataset, frac_train: float = 0.8, frac_valid: float = 0.1, frac_test: float = 0.1,
              seed: int = None, **kwargs) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits a dataset into train/validation/test using a stratified split.

        Parameters
        ----------
        dataset: Dataset
            Dataset to split
        frac_train: float
            Fraction of dataset to use for training
        frac_valid: float
            Fraction of dataset to use for validation
        frac_test: float
            Fraction of dataset to use for testing
        seed: int
            Seed for the random number generator
        kwargs

        Returns
        -------
        train_indexes: List[int]
            Indexes of the training set
        valid_indexes: List[int]
            Indexes of the validation set
        test_indexes: List[int]
            Indexes of the test set
        """

        if seed is not None:
            np.random.seed(seed)

        if frac_valid == 0:
            stratifier = IterativeStratification(n_splits=2, order=1,
                                                 sample_distribution_per_fold=[frac_test, frac_train])
            train_indexes, test_indexes = next(stratifier.split(dataset.smiles, dataset.y))

            return train_indexes, [], test_indexes
        else:
            stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[frac_test,
                                                                                                    1 - frac_test])
            train_indexes, test_indexes = next(stratifier.split(dataset.smiles, dataset.y))

            new_frac_train = frac_train / (1 - frac_test)
            stratifier = IterativeStratification(n_splits=2, order=1,
                                                 sample_distribution_per_fold=[1 - new_frac_train, new_frac_train])

            new_train_indexes, valid_indexes = next(stratifier.split(dataset.smiles[train_indexes],
                                                                     dataset.y[train_indexes]))

            new_train_indexes = train_indexes[new_train_indexes]
            valid_indexes = train_indexes[valid_indexes]
            train_indexes = new_train_indexes

            return train_indexes, valid_indexes, test_indexes

    def k_fold_split(self, dataset: Dataset, k: int, seed: int = None) -> List[Tuple[Dataset, Dataset]]:

        """
        Split the dataset into k folds using stratified sampling.

        Parameters
        ----------
        dataset: Dataset
            The dataset to split.
        k: int
            The number of folds.
        seed:
            The seed to use for the random number generator.

        Returns
        -------
        folds: List[Tuple[Dataset, Dataset]]
            A list of tuples (train_dataset, test_dataset) containing the k folds.
        """

        stratifier = IterativeStratification(n_splits=k, order=1)
        folds = []
        for fold in stratifier.split(dataset.smiles, dataset.y):
            train_indexes, test_indexes = fold
            train_dataset = dataset.select_to_split(train_indexes)
            test_dataset = dataset.select_to_split(test_indexes)
            folds.append((train_dataset, test_dataset))

        return folds
