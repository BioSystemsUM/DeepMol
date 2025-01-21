

import copy
from typing import List, Tuple

import numpy as np
from deepmol.datasets.datasets import Dataset
from deepmol.splitters.splitters import Splitter

from graph_part.molecules import stratified_k_fold, train_test_validation_split


class GraphPartSimilaritySplitter(Splitter):
    """
    Class for doing data splits by stratification on a single task.
    """

    def __init__(self, stratified=False, threshold=0.3):
        self.stratified = stratified
        self.threshold = threshold
        super().__init__()

    def k_fold_split(self,
                     dataset: Dataset,
                     k: int,
                     seed: int = None) -> List[Tuple[Dataset, Dataset]]:
        """
        Splits compounds into k-folds using stratified sampling.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        k: int
            Number of folds to split `dataset` into.
        seed: int
            Random seed to use.

        Returns
        -------
        fold_datasets: List[Tuple[NumpyDataset, NumpyDataset]]:
            A list of length k of tuples of train and test datasets as NumpyDataset objects.
        """
        raise NotImplementedError

    def split(self,
              dataset: Dataset,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              **kwargs) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits compounds into train/validation/test using stratified sampling.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        frac_train: float
            Fraction of dataset put into training data.
        frac_valid: float
            Fraction of dataset put into validation data.
        frac_test: float
            Fraction of dataset put into test data.
        seed: int
            Random seed to use.
        force_split: bool
            If True, will force the split without checking if it is a regression or classification label.

        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            A tuple of train indices, valid indices, and test indices.
        """
        np.testing.assert_equal(frac_train + frac_valid + frac_test, 1.)
        np.testing.assert_equal(10 * frac_train + 10 * frac_valid + 10 * frac_test, 10.)

        frac_train = round(frac_train, 2)
        frac_test = round(frac_test, 2)
        frac_valid = round(frac_valid, 2)

        # divide idx by y value
        if self.stratified:
            indexes = train_test_validation_split(dataset.smiles, labels=dataset.y, threshold=self.threshold, test_size=frac_test, valid_size=frac_valid)
        else:
            indexes = train_test_validation_split(dataset.smiles, threshold=self.threshold, test_size=frac_test, valid_size=frac_valid)

        if len(indexes) == 2:
            if frac_valid == 0:
                valid_indexes = list(map(int, []))
                test_indexes = list(map(int, indexes[1]))
            elif frac_test == 0:
                test_indexes = list(map(int, []))
                valid_indexes = list(map(int, indexes[1]))
        
        else:
            test_indexes = list(map(int, indexes[1]))
            valid_indexes = list(map(int, indexes[2]))
            
        train_indexes = list(map(int, indexes[0]))
        return train_indexes, valid_indexes, test_indexes