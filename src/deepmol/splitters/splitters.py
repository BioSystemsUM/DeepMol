import copy
from abc import abstractmethod, ABC

import numpy as np

from typing import Tuple, List

from rdkit import DataStructs
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.ML.Cluster import Butina

from deepmol.datasets import Dataset
from sklearn.model_selection import KFold, StratifiedKFold

from deepmol.loggers.logger import Logger
from deepmol.splitters._utils import get_train_valid_test_indexes, get_fingerprints_for_each_class, \
    get_mols_for_each_class


class Splitter(ABC):
    """
    Splitters split up datasets into pieces for training/validation/testing.
    In machine learning applications, it's often necessary to split up a dataset into training/validation/test sets.
    Or to k-fold split a dataset for cross-validation.
    """

    def __init__(self):
        self.logger = Logger()

    def train_valid_test_split(self,
                               dataset: Dataset,
                               frac_train: float = 0.8,
                               frac_valid: float = None,
                               frac_test: float = None,
                               seed: int = None,
                               **kwargs) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Splits a Dataset into train/validation/test sets.
        Returns Dataset objects for train, valid, test.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        frac_train: float
            The fraction of data to be used for the training split.
        frac_valid: float
            The fraction of data to be used for the validation split.
        frac_test: float
            The fraction of data to be used for the test split.
        seed: int
            Random seed to use.
        **kwargs: Dict[str, Any]
            Other arguments.

        Returns
        -------
        Tuple[Dataset, Dataset, Dataset]
          A tuple of train, valid and test datasets as Dataset objects.
        """

        if frac_test is None and frac_valid is None:
            raise Exception("Please insert all the required parameters! Both test and validation sets fraction are "
                            "not defined!")

        elif frac_test is None:
            frac_test = 1 - (frac_train + frac_valid)

        elif frac_valid is None:
            frac_valid = 1 - (frac_train + frac_test)

        train_inds, valid_inds, test_inds = self.split(dataset,
                                                       frac_train=frac_train,
                                                       frac_test=frac_test,
                                                       frac_valid=frac_valid,
                                                       seed=seed,
                                                       **kwargs)

        train_dataset = dataset.select_to_split(train_inds)
        valid_dataset = dataset.select_to_split(valid_inds)
        test_dataset = dataset.select_to_split(test_inds)
        if isinstance(train_dataset, Dataset):
            train_dataset.memory_cache_size = 40 * (1 << 20)  # 40 MB

        return train_dataset, valid_dataset, test_dataset

    def train_test_split(self,
                         dataset: Dataset,
                         frac_train: float = 0.8,
                         seed: int = None,
                         **kwargs) -> Tuple[Dataset, Dataset]:
        """
        Splits self into train/test sets.
        Returns Dataset objects for train/test.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        frac_train: float
            The fraction of data to be used for the training split.
        seed: int
            Random seed to use.
        **kwargs: Dict[str, Any]
            Other arguments.

        Returns
        -------
        Tuple[Dataset, Dataset]
          A tuple of train and test datasets as Dataset objects.
        """
        train_dataset, _, test_dataset = self.train_valid_test_split(dataset,
                                                                     frac_train=frac_train,
                                                                     frac_test=1 - frac_train,
                                                                     frac_valid=0.,
                                                                     seed=seed,
                                                                     **kwargs)
        return train_dataset, test_dataset

    @abstractmethod
    def split(self,
              dataset: Dataset,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              seed: int = None,
              **kwargs) -> Tuple[List[int], List[int], List[int]]:
        """
        Return indices for specified splits.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        frac_train: float
            The fraction of data to be used for the training split.
        frac_valid: float
            The fraction of data to be used for the validation split.
        frac_test: float
            The fraction of data to be used for the test split.
        seed: int
            Random seed to use.
        **kwargs: Dict[str, Any]
            Other arguments.

        Returns
        -------
        Tuple[List[int], List[int], List[int]]
          A tuple `(train_inds, valid_inds, test_inds)` of the indices for the various splits.
        """

    @abstractmethod
    def k_fold_split(self,
                     dataset: Dataset,
                     k: int,
                     seed: int = None) -> List[Tuple[Dataset, Dataset]]:
        """
        Split a dataset into k folds for cross-validation.

        Parameters
        ----------
        dataset: Dataset
            Dataset to do a k-fold split
        k: int
            Number of folds to split `dataset` into.
        seed: int, optional
            Random seed to use for reproducibility.

        Returns
        -------
        List[Tuple[Dataset, Dataset]]
          List of length k tuples of (train, test) where `train` and `test` are both `Dataset`.
        """


class RandomSplitter(Splitter):
    """
    Class for doing random data splits.
    """

    def split(self,
              dataset: Dataset,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              seed: int = None,
              **kwargs) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits randomly into train/validation/test.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        frac_train: float
            The fraction of data to be used for the training split.
        frac_valid: float
            The fraction of data to be used for the validation split.
        frac_test: float
            The fraction of data to be used for the test split.
        seed: int
            Random seed to use.
        **kwargs: Dict[str, Any]
            Other arguments.

        Returns
        -------
        Tuple[List[int], List[int], List[int]]
          A tuple of train indices, valid indices, and test indices.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
        if seed is not None:
            np.random.seed(seed)
        num_datapoints = dataset.__len__()
        train_cutoff = int(frac_train * num_datapoints)
        valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
        shuffled = np.random.permutation(range(num_datapoints))
        return list(shuffled[:train_cutoff]), list(shuffled[train_cutoff:valid_cutoff]), list(shuffled[valid_cutoff:])

    def k_fold_split(self,
                     dataset: Dataset,
                     k: int,
                     seed: int = None) -> List[Tuple[Dataset, Dataset]]:
        """
        Split a dataset into k folds for cross-validation.

        Parameters
        ----------
        dataset: Dataset
            Dataset to do a k-fold split
        k: int
            Number of folds to split `dataset` into.
        seed: int, optional
            Random seed to use for reproducibility.

        Returns
        -------
        List[Tuple[Dataset, Dataset]]
          List of length k tuples of (train, test) where `train` and `test` are both `Dataset`.
        """
        self.logger.info("Computing K-fold split")
        ds = copy.deepcopy(dataset)

        kf = KFold(n_splits=k, shuffle=True, random_state=seed)

        train_datasets = []
        test_datasets = []
        for train_index, test_index in kf.split(ds.X):
            train_datasets.append(ds.select_to_split(train_index))
            test_datasets.append(ds.select_to_split(test_index))

        return list(zip(train_datasets, test_datasets))


class SingletaskStratifiedSplitter(Splitter):
    """
    Class for doing data splits by stratification on a single task.
    """

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
        self.logger.info("Computing Stratified K-fold split")
        ds = copy.deepcopy(dataset)

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

        train_datasets = []
        test_datasets = []
        for train_index, test_index in skf.split(ds.X, ds.y):
            train_datasets.append(ds.select_to_split(train_index))
            test_datasets.append(ds.select_to_split(test_index))

        return list(zip(train_datasets, test_datasets))

    def split(self,
              dataset: Dataset,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              seed: int = None,
              force_split: bool = False,
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

        if seed is not None:
            np.random.seed(seed)

        train_idx = np.array([])
        valid_idx = np.array([])
        test_idx = np.array([])

        # divide idx by y value

        if not force_split:
            # check if regression or classification (assume regression if there are more than 10 unique y values)
            classes = np.all(np.isclose(dataset.y, np.round(dataset.y), equal_nan=True))
            if not classes:
                raise ValueError("Cannot stratify by regression labels. Use other splitter instead. "
                                 "If you want to force the split, set force_split=True.")
        remaining_idx = []
        idx_by_class = {}
        classes = np.unique(dataset.y)
        for c in classes:
            idx_by_class[c] = np.where(dataset.y == c)[0]
            np.random.shuffle(idx_by_class[c])
            train_stop = int(frac_train * len(idx_by_class[c]))
            valid_stop = int(train_stop + (frac_valid * len(idx_by_class[c])))
            test_stop = int(valid_stop + (frac_test * len(idx_by_class[c])))
            train_idx = np.hstack([train_idx, idx_by_class[c][:train_stop]])
            valid_idx = np.hstack([valid_idx, idx_by_class[c][train_stop:valid_stop]])
            test_idx = np.hstack([test_idx, idx_by_class[c][valid_stop:test_stop]])
            remaining_idx.extend(idx_by_class[c][test_stop:])

        # divide remaining idx randomly by test, valid and test (according to frac_test, frac_valid, frac_test)
        if len(remaining_idx) > 0:
            np.random.shuffle(remaining_idx)
            train_remaining = int(frac_train * len(dataset.y)) - len(train_idx)
            train_idx = np.hstack([train_idx, remaining_idx[:train_remaining]])
            valid_remaining = int(frac_valid * len(dataset.y)) - len(valid_idx)
            valid_idx = np.hstack([valid_idx, remaining_idx[train_remaining:train_remaining + valid_remaining]])
            test_idx = np.hstack([test_idx, remaining_idx[train_remaining + valid_remaining:]])

        train_indexes = list(map(int, train_idx))
        valid_indexes = list(map(int, valid_idx))
        test_indexes = list(map(int, test_idx))
        return train_indexes, valid_indexes, test_indexes


class SimilaritySplitter(Splitter):
    """
    Class for doing data splits based on fingerprint similarity.
    """

    def split(self,
              dataset: Dataset,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              seed: int = None,
              homogenous_threshold: float = 0.7) -> Tuple[List[int], List[int], List[int]]:

        """
        Splits compounds into train/validation/test based on similarity.
        It can generate both homogenous and heterogeneous train and test sets.

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
        homogenous_threshold: float
            Threshold for similarity, all the compounds with a similarity lower than this threshold will be separated
            in the training set and test set. The higher the threshold is, the more heterogeneous the split will be.

        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            A tuple of train indices, valid indices, and test indices.
        """
        fps_classes_map, indices_classes_map, all_fps = get_fingerprints_for_each_class(dataset)

        train_size = int(frac_train * len(dataset))
        valid_size = int(frac_valid * len(dataset))
        test_size = len(dataset) - train_size - valid_size

        train_inds = []
        test_valid_inds = []

        is_regression = dataset.mode == 'regression'

        if not is_regression:
            for class_ in fps_classes_map:
                train_size_class_ = int(frac_train * len(fps_classes_map[class_]))
                valid_size_class_ = int(frac_valid * len(fps_classes_map[class_]))
                test_size_class_ = len(fps_classes_map[class_]) - train_size_class_ - valid_size_class_

                fps = fps_classes_map[class_]
                indexes = indices_classes_map[class_]

                train_inds_class_, test_valid_inds_class_ = self._split_fingerprints(fps, train_size_class_,
                                                                                     valid_size_class_ +
                                                                                     test_size_class_,
                                                                                     indexes,
                                                                                     homogenous_threshold)

                train_inds.extend(train_inds_class_)
                test_valid_inds.extend(test_valid_inds_class_)
        else:
            train_inds, test_valid_inds = self._split_fingerprints(all_fps, train_size,
                                                                   valid_size +
                                                                   test_size,
                                                                   [i for i in range(len(all_fps))],
                                                                   homogenous_threshold)

        # Split the second group into validation and test sets.

        if valid_size == 0:
            valid_inds = []
            test_inds = test_valid_inds
        elif test_size == 0:
            test_inds = []
            valid_inds = test_valid_inds
        else:

            test_inds = []
            valid_inds = []
            if not is_regression:
                for class_ in fps_classes_map:
                    indexes = [i for i in indices_classes_map[class_] if i in test_valid_inds]
                    test_valid_fps_class_ = [all_fps[i] for i in indexes]

                    train_size_class_ = int(frac_train * len(fps_classes_map[class_]))
                    valid_size_class_ = int(frac_valid * len(fps_classes_map[class_]))
                    test_size_class_ = len(fps_classes_map[class_]) - train_size_class_ - valid_size_class_

                    test_inds_class_, valid_inds_class_ = self._split_fingerprints(test_valid_fps_class_,
                                                                                   test_size_class_,
                                                                                   valid_size_class_,
                                                                                   indexes, homogenous_threshold)

                    test_inds.extend(test_inds_class_)
                    valid_inds.extend(valid_inds_class_)

            else:
                test_valid_idx = [i for i in range(len(all_fps)) if i in test_valid_inds]
                test_valid_fps = [all_fps[i] for i in test_valid_idx]
                test_inds, valid_inds = self._split_fingerprints(test_valid_fps,
                                                                 test_size,
                                                                 valid_size,
                                                                 test_valid_inds,
                                                                 homogenous_threshold)

        return train_inds, valid_inds, test_inds

    def k_fold_split(self,
                     dataset: Dataset,
                     k: int,
                     seed: int = None) -> List[Tuple[Dataset, Dataset]]:
        """
        Splits the dataset into k folds based on similarity.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        k: int
            Number of folds.
        seed: int
            Random seed.

        Returns
        -------
        List[Tuple[Dataset, Dataset]]
            List of train/test pairs of size k.
        """

    @staticmethod
    def _split_fingerprints(fps: List, size1: int, size2: int, indexes: List[int], homogenous_threshold: float):
        """
        Returns all scaffolds from the dataset.

        Parameters
        ----------
        fps: List
            List of fingerprints
        size1: int
            Size of the first set of molecules
        size2: int
            Size of the second set of molecules
        indexes: List[int]
            Molecules' indexes
        homogenous_threshold:
            Threshold for similarity, all the compounds with a similarity lower than this threshold will be separated
            from the training set and test set

        Returns
        -------
        scaffold_sets: List[List[int]]
            List of indices of each scaffold in the dataset.
        """
        assert len(fps) == size1 + size2

        fp_in_group = [[fps[0]], []]

        indices_in_group: Tuple[List[int], List[int]] = ([0], [])
        remaining_fp = fps[1:]
        remaining_indices = indexes[1:]
        max_similarity_to_group = [
            DataStructs.BulkTanimotoSimilarity(fps[0], remaining_fp),
            [0] * len(remaining_fp)
        ]
        while len(remaining_fp) > 0:
            # Decide which group to assign a molecule to.

            if len(fp_in_group[0]) / size1 <= len(fp_in_group[1]) / size2:
                group = 0
            else:
                group = 1

            # Identify the unassigned molecule that is least similar to everything in
            # the other group.
            minimum = np.min(max_similarity_to_group[1 - group])
            if minimum < homogenous_threshold:
                i = np.argmin(max_similarity_to_group[1 - group])

            else:
                # list_elements = np.array([i for i in range(len(max_similarity_to_group[1 - group]))])
                i = np.argmax(max_similarity_to_group[1 - group])
                # i = np.random.choice(list_elements)

            # Add it to the group.

            fp = remaining_fp[i]
            fp_in_group[group].append(fp)
            indices_in_group[group].append(remaining_indices[i])

            # Update the data on unassigned molecules.

            similarity = DataStructs.BulkTanimotoSimilarity(fp, remaining_fp)
            max_similarity_to_group[group] = np.delete(
                np.maximum(similarity, max_similarity_to_group[group]), i)
            max_similarity_to_group[1 - group] = np.delete(
                max_similarity_to_group[1 - group], i)
            del remaining_fp[i]
            del remaining_indices[i]

        return indices_in_group


class ScaffoldSplitter(Splitter):
    """
    Class for splitting the dataset based on scaffolds.
    """

    def split(self,
              dataset: Dataset,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              seed: int = None,
              homogenous_datasets: bool = True) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits internal compounds into train/validation/test by scaffold.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        frac_train: float
            The fraction of data to be used for the training split.
        frac_valid: float
            The fraction of data to be used for the validation split.
        frac_test: float
            The fraction of data to be used for the test split.
        seed: int
            Random seed to use.
        homogenous_datasets: bool
            Whether the datasets will be homogenous or not.

        Returns
        -------
        Tuple[List[int], List[int], List[int]]
          A tuple of train indices, valid indices, and test indices.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)

        is_regression = dataset.mode == 'regression'

        train_cutoff = int(frac_train * len(dataset))
        valid_cutoff = int(frac_valid * len(dataset))
        test_cutoff = len(dataset) - train_cutoff - valid_cutoff

        train_inds: List[int] = []
        valid_inds: List[int] = []
        test_inds: List[int] = []

        if not is_regression:
            mols_classes_map, indices_classes_map = get_mols_for_each_class(dataset)
            for class_ in mols_classes_map:
                mols = mols_classes_map[class_]
                indexes = indices_classes_map[class_]
                scaffold_sets = self.generate_scaffolds(mols, indexes)
                train_class_cutoff = int(frac_train * len(indexes))
                valid_class_cutoff = int(frac_valid * len(indexes))
                test_class_cutoff = len(indexes) - train_class_cutoff - valid_class_cutoff

                train_inds_class_, valid_inds_class_, test_inds_class_ = \
                    get_train_valid_test_indexes(scaffold_sets, train_class_cutoff,
                                                 test_class_cutoff, valid_class_cutoff,
                                                 frac_train, frac_test, homogenous_datasets)

                train_inds.extend(train_inds_class_)
                test_inds.extend(test_inds_class_)
                valid_inds.extend(valid_inds_class_)

        else:
            idsx = [i for i in range(len(dataset.mols))]
            scaffold_sets = self.generate_scaffolds(dataset.mols, idsx)
            train_inds, valid_inds, test_inds = get_train_valid_test_indexes(scaffold_sets,
                                                                             train_cutoff,
                                                                             test_cutoff,
                                                                             valid_cutoff,
                                                                             frac_train,
                                                                             frac_test,
                                                                             homogenous_datasets)

        return train_inds, valid_inds, test_inds

    def k_fold_split(self,
                     dataset: Dataset,
                     k: int,
                     seed: int = None) -> List[Tuple[Dataset, Dataset]]:
        """
        Splits the dataset into k folds based on scaffolds.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        k: int
            Number of folds.
        seed: int
            Random seed.

        Returns
        -------
        List[Tuple[Dataset, Dataset]]
            List of train/test pairs of size k.
        """

    @staticmethod
    def generate_scaffolds(mols: np.ndarray,
                           indexes: List[int]) -> List[List[int]]:
        """
        Returns all scaffolds from the dataset.

        Parameters
        ----------
        mols: List[Mol]
            List of rdkit Mol objects for scaffold generation
        indexes: List[int]
            Molecules' indexes.

        Returns
        -------
        scaffold_sets: List[List[int]]
            List of indices of each scaffold in the dataset.
        """
        scaffolds = {}
        for ind, mol in enumerate(mols):
            scaffold = ScaffoldSplitter._generate_scaffold(mol)

            if scaffold not in scaffolds:
                scaffolds[scaffold] = [indexes[ind]]
            else:
                scaffolds[scaffold].append(indexes[ind])

        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}

        scaffold_scaffold_set = sorted(scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)

        scaffold_sets = [scaffold_set for _, scaffold_set in scaffold_scaffold_set]

        return scaffold_sets

    @staticmethod
    def _generate_scaffold(mol: Mol, include_chirality: bool = False) -> str:
        """
        Compute the Bemis-Murcko scaffold for a SMILES string.
        Bemis-Murcko scaffolds are described in DOI: 10.1021/jm9602928.
        They are essentially that part of the molecule consisting of rings and the linker atoms between them.

        Parameters
        ---------
        mol: Mol
            rdkit Mol object for scaffold generation
        include_chirality: bool
            Whether to include chirality in scaffolds or not.

        Returns
        -------
        scaffold: str
            The MurckScaffold SMILES from the original SMILES
        """
        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold


class ButinaSplitter(Splitter):
    """
    Splitter based on the Butina clustering algorithm.
    """

    def __init__(self, cutoff: float = 0.6):
        """
        Create a ButinaSplitter.

        Parameters
        ----------
        cutoff: float
            The cutoff value for tanimoto similarity.  Molecules that are more similar than this will tend to be put in
            the same dataset.
        """
        super().__init__()
        self.cutoff = cutoff

    def split(self,
              dataset: Dataset,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              seed: int = None,
              homogenous_datasets: bool = True) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits internal compounds into train and validation based on the butina clustering algorithm. The dataset is
        expected to be a classification dataset.
        This algorithm is designed to generate validation data that are novel chemotypes.
        Setting a small cutoff value will generate smaller, finer clusters of high similarity, whereas setting a large
        cutoff value will generate larger, coarser clusters of low similarity.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        frac_train: float
            The fraction of data to be used for the training split.
        frac_valid: float
            The fraction of data to be used for the validation split.
        frac_test: float
            The fraction of data to be used for the test split.
        seed: int
            Random seed to use.
        homogenous_datasets: bool
            Whether the datasets will be homogenous or not.

        Returns
        -------
        Tuple[List[int], List[int], List[int]]
          A tuple of train indices, valid indices, and test indices.
        """
        fps_classes_map, indices_classes_map, all_fps = get_fingerprints_for_each_class(dataset)

        is_regression = dataset.mode == 'regression'

        train_cutoff = int(frac_train * len(dataset))
        valid_cutoff = int(frac_valid * len(dataset))
        test_cutoff = len(dataset) - train_cutoff - valid_cutoff

        train_inds: List[int] = []
        valid_inds: List[int] = []
        test_inds: List[int] = []

        if not is_regression:
            for class_ in fps_classes_map:
                dists = []
                fps = fps_classes_map[class_]
                nfps = len(fps)
                for i in range(1, nfps):
                    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
                    dists.extend([1 - x for x in sims])
                scaffold_sets = Butina.ClusterData(
                    dists, nfps, self.cutoff, isDistData=True)
                scaffold_sets = sorted(scaffold_sets, key=lambda x: -len(x))

                new_scaffold_sets = []  # update for the true indexes of the compounds
                for scaffold_list in scaffold_sets:
                    new_scaffold_set = []
                    for i in scaffold_list:
                        true_index = indices_classes_map[class_][i]
                        new_scaffold_set.append(true_index)
                    new_scaffold_sets.append(new_scaffold_set)

                train_cutoff_class_ = int(frac_train * len(fps_classes_map[class_]))
                valid_cutoff_class_ = int(frac_valid * len(fps_classes_map[class_]))
                test_cutoff_class_ = len(fps_classes_map[class_]) - train_cutoff_class_ - valid_cutoff_class_

                train_inds_class_, valid_inds_class_, test_inds_class_ = \
                    get_train_valid_test_indexes(new_scaffold_sets, train_cutoff_class_, test_cutoff_class_,
                                                 valid_cutoff_class_, frac_train, frac_test, homogenous_datasets)

                train_inds.extend(train_inds_class_)
                test_inds.extend(test_inds_class_)
                valid_inds.extend(valid_inds_class_)

        else:
            dists = []
            nfps = len(all_fps)
            for i in range(1, nfps):
                sims = DataStructs.BulkTanimotoSimilarity(all_fps[i], all_fps[:i])
                dists.extend([1 - x for x in sims])
            scaffold_sets = Butina.ClusterData(
                dists, nfps, self.cutoff, isDistData=True)
            scaffold_sets = sorted(scaffold_sets, key=lambda x: -len(x))
            counter = 0
            for scaffold_set in scaffold_sets:
                counter += len(scaffold_set)

            train_inds, valid_inds, test_inds = get_train_valid_test_indexes(scaffold_sets,
                                                                             train_cutoff, test_cutoff, valid_cutoff,
                                                                             frac_train, frac_test, homogenous_datasets)

        return train_inds, valid_inds, test_inds

    def k_fold_split(self,
                     dataset: Dataset,
                     k: int,
                     seed: int = None) -> List[Tuple[Dataset, Dataset]]:
        """
        Splits the dataset into k folds based on Butina splitter.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        k: int
            Number of folds.
        seed: int
            Random seed.

        Returns
        -------
        List[Tuple[Dataset, Dataset]]
            List of train/test pairs of size k.
        """
