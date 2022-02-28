from abc import abstractmethod, ABC

import numpy as np

from typing import Tuple, List, Optional, Type

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Mol, MolFromSmiles

from datasets.datasets import Dataset, NumpyDataset

from sklearn.model_selection import KFold, StratifiedKFold


def get_train_valid_test_indexes(scaffold_sets,
                                 train_cutoff, test_cutoff, valid_cutoff,
                                 frac_train, frac_test,
                                 homogenous_datasets):
    train_inds = np.array([])
    valid_inds = np.array([])
    test_inds = np.array([])

    for scaffold_set in scaffold_sets:
        scaffold_set = np.array(scaffold_set)
        if not homogenous_datasets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(test_inds) + len(scaffold_set) <= test_cutoff:
                    test_inds = np.hstack([test_inds, scaffold_set])
                elif len(valid_inds) + len(scaffold_set) <= valid_cutoff:
                    valid_inds = np.hstack([valid_inds, scaffold_set])
                else:
                    np.random.shuffle(scaffold_set)
                    train_index = int(np.round(len(scaffold_set) * frac_train))
                    test_index = int(np.round(len(scaffold_set) * frac_test)) + train_index
                    train_inds = np.hstack([train_inds, scaffold_set[:train_index]])
                    test_inds = np.hstack([test_inds, scaffold_set[train_index:test_index]])
                    valid_inds = np.hstack([valid_inds, scaffold_set[test_index:]])
            else:
                train_inds = np.hstack([train_inds, scaffold_set])

        else:
            np.random.shuffle(scaffold_set)
            train_index = int(np.round(len(scaffold_set) * frac_train))
            test_index = int(np.round(len(scaffold_set) * frac_test)) + train_index
            train_inds = np.hstack([train_inds, scaffold_set[:train_index]])
            test_inds = np.hstack([test_inds, scaffold_set[train_index:test_index]])
            valid_inds = np.hstack([valid_inds, scaffold_set[test_index:]])

    return list(map(int, train_inds)), list(map(int, valid_inds)), list(map(int, test_inds))


def get_mols_for_each_class(mols: List[Mol], dataset: Dataset):
    mols_classes_map = {}
    indices_classes_map = {}
    for i, mol in enumerate(mols):

        if dataset.y[i] not in mols_classes_map:
            mols_classes_map[dataset.y[i]] = [mol]
            indices_classes_map[dataset.y[i]] = [i]

        else:
            mols_classes_map[dataset.y[i]].append(mol)
            indices_classes_map[dataset.y[i]].append(i)

    return mols_classes_map, indices_classes_map


def get_fingerprints_for_each_class(mols, dataset):
    fps_classes_map = {}
    indices_classes_map = {}
    all_fps = []
    for i, mol in enumerate(mols):

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
        all_fps.append(fp)
        if dataset.y[i] not in fps_classes_map:
            fps_classes_map[dataset.y[i]] = [fp]
            indices_classes_map[dataset.y[i]] = [i]

        else:
            fps_classes_map[dataset.y[i]].append(fp)
            indices_classes_map[dataset.y[i]].append(i)

    return fps_classes_map, indices_classes_map, all_fps


class Splitter(ABC):
    """Splitters split up datasets into pieces for training/validation/testing.
    In machine learning applications, it's often necessary to split up a dataset
    into training/validation/test sets. Or to k-fold split a dataset for cross-validation.
    """

    # TODO: Possible upgrade: add directories input to save splits to file (update code)
    def k_fold_split(self,
                     dataset: Dataset,
                     k: int,
                     seed: Optional[int] = None,  # added
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
            train_datasets.append(ds.select_to_split(train_index))
            test_datasets.append(ds.select_to_split(test_index))

        return list(zip(train_datasets, test_datasets))

    def train_valid_test_split(self,
                               dataset: Dataset,
                               frac_train: float = 0.8,
                               frac_valid: float = None,
                               frac_test: float = None,
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

        if frac_test is None and frac_valid is None:
            raise Exception("Please insert all the required parameters! Both test and validation sets fraction are "
                            "not defined!")

        elif frac_test is None:
            frac_test = 1 - frac_train + frac_valid

        elif frac_valid is None:
            frac_valid = 1 - frac_train + frac_test

        # print("Computing train/valid/test indices")
        train_inds, valid_inds, test_inds = self.split(dataset,
                                                       frac_train=frac_train,
                                                       frac_test=frac_test,
                                                       frac_valid=frac_valid,
                                                       seed=seed,
                                                       log_every_n=log_every_n,
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

    @abstractmethod
    def split(self,
              dataset: Dataset,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              seed: Optional[int] = None,
              log_every_n: Optional[int] = None,
              **kwargs) -> Tuple:

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

    def split(self, dataset: Type[Dataset],
              frac_train: float = 0.8, frac_valid: float = 0.1,
              frac_test: float = 0.1,
              seed: Optional[int] = None,
              log_every_n: Optional[int] = None,
              **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        # num_datapoints = len(dataset)
        num_datapoints = dataset.len_mols()
        train_cutoff = int(frac_train * num_datapoints)
        valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
        shuffled = np.random.permutation(range(num_datapoints))
        return shuffled[:train_cutoff], shuffled[train_cutoff:valid_cutoff], shuffled[valid_cutoff:]


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
                     **kwargs) -> List[Tuple[NumpyDataset, NumpyDataset]]:
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

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)  # changed so that users can define the seed

        train_datasets = []
        test_datasets = []
        for train_index, test_index in skf.split(ds.X, ds.y):
            train_datasets.append(ds.select_to_split(train_index))
            test_datasets.append(ds.select_to_split(test_index))

        return list(zip(train_datasets, test_datasets))

    def split(self, dataset: Dataset, frac_train: float = 0.8,
              frac_valid: float = 0.1, frac_test: float = 0.1,
              seed: Optional[int] = None,
              log_every_n: Optional[int] = None,
              **kwargs) -> Tuple[List[int], List[int], List[int]]:
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

        return list(map(int, train_idx)), list(map(int, valid_idx)), list(map(int, test_idx))


class SimilaritySplitter(Splitter):

    def split(self, dataset: Dataset, frac_train: float = 0.8,
              frac_valid: float = 0.1, frac_test: float = 0.1,
              seed: Optional[int] = None, log_every_n: Optional[int] = None,
              homogenous_threshold: float = 0.7) -> Tuple:

        """
        Splits compounds into train/validation/test based on similarity.
        It can generate both homogenous and heterogeneous train and test sets
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
        homogenous_threshold: float
          Threshold for similarity, all the compounds with a similarity lower
           than this threshold will be separated in the training set and test set.
          The higher the threshold is, the more heterogeneous the split will be.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
          A tuple of train indices, valid indices, and test indices.
          Each indices is a numpy array.
                """

        mols = generate_mols_and_delete_invalid_smiles(dataset)

        fps_classes_map, indices_classes_map, all_fps = get_fingerprints_for_each_class(mols, dataset)

        train_size = int(frac_train * len(dataset))
        valid_size = int(frac_valid * len(dataset))
        test_size = len(dataset) - train_size - valid_size

        train_inds = []
        test_valid_inds = []

        is_regression = any([not isinstance(i.item(), int) and not i.item().is_integer() for i in dataset.y])

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
                test_valid_fps = [i for i in range(len(all_fps)) if i in test_valid_inds]
                test_inds, valid_inds = self._split_fingerprints(test_valid_fps,
                                                                 test_size,
                                                                 valid_size,
                                                                 test_valid_inds,
                                                                 homogenous_threshold)

        return train_inds, valid_inds, test_inds

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
          Threshold for similarity, all the compounds with a similarity lower
           than this threshold will be separated from the training set and test set
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

    def split(self,
              dataset: Dataset,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              seed: Optional[int] = None,
              log_every_n: Optional[int] = 1000,
              homogenous_datasets: bool = True) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits internal compounds into train/validation/test by scaffold.
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
        homogenous_datasets: bool, optional (default True)
          Whether the datasets will be homogenous or not.
        Returns
        -------
        Tuple[List[int], List[int], List[int]]
          A tuple of train indices, valid indices, and test indices.
          Each indices is a list of integers.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)

        mols = generate_mols_and_delete_invalid_smiles(dataset)
        mols_classes_map, indices_classes_map = get_mols_for_each_class(mols, dataset)

        is_regression = any([not isinstance(i.item(), int) and not i.item().is_integer() for i in dataset.y])

        train_cutoff = int(frac_train * len(dataset))
        valid_cutoff = int(frac_valid * len(dataset))
        test_cutoff = len(dataset) - train_cutoff - valid_cutoff

        train_inds: List[int] = []
        valid_inds: List[int] = []
        test_inds: List[int] = []

        if not is_regression:
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
            scaffold_sets = self.generate_scaffolds(mols, [i for i in range(len(mols))])
            train_inds, valid_inds, test_inds = get_train_valid_test_indexes(scaffold_sets, train_cutoff, test_cutoff,
                                                                             valid_cutoff, frac_train, frac_test,
                                                                             homogenous_datasets)

        return train_inds, valid_inds, test_inds

    @staticmethod
    def generate_scaffolds(mols: List[Mol], indexes: List[int],
                           log_every_n: int = 1000) -> List[List[int]]:
        """Returns all scaffolds from the dataset.
        Parameters
        ----------
        mols: List[Mol]
          List of rdkit.Mol objects for scaffold generation
        indexes: List[int]
          Molecules' indexes.
        log_every_n: int, optional (default 1000)
          Controls the logger by dictating how often logger outputs
          will be produced.
        Returns
        -------
        scaffold_sets: List[List[int]]
          List of indices of each scaffold in the dataset.
        """
        scaffolds = {}
        data_len = len(mols)
        for ind, mol in enumerate(mols):
            if ind % log_every_n == 0:
                print("Generating scaffold %d/%d" % (ind, data_len))

            if isinstance(mol, str):
                try:
                    mol_object = MolFromSmiles(mol)
                    scaffold = ScaffoldSplitter._generate_scaffold(mol_object)
                except:
                    scaffold = None

            elif isinstance(mol, Mol):
                scaffold = ScaffoldSplitter._generate_scaffold(mol)

            else:
                scaffold = False

            if scaffold is not None:
                if scaffold not in scaffolds:
                    scaffolds[scaffold] = [indexes[ind]]
                else:
                    scaffolds[scaffold].append(indexes[ind])

        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}

        scaffold_sets = []

        scaffold_scaffold_set = sorted(scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)

        for scaffold, scaffold_set in scaffold_scaffold_set:
            scaffold_sets.append(scaffold_set)

        return scaffold_sets

    @staticmethod
    def _generate_scaffold(mol: Mol, include_chirality: bool = False) -> str:
        """Compute the Bemis-Murcko scaffold for a SMILES string.
        Bemis-Murcko scaffolds are described in DOI: 10.1021/jm9602928.
        They are essentially that part of the molecule consisting of
        rings and the linker atoms between them.
        Paramters
        ---------
        smiles: str
          SMILES
        include_chirality: bool, default False
          Whether to include chirality in scaffolds or not.
        Returns
        -------
        str
          The MurckScaffold SMILES from the original SMILES
        References
        ----------
        .. [1] Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
           1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.
        Note
        ----
        This function requires RDKit to be installed.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
        except ModuleNotFoundError:
            raise ImportError("This function requires RDKit to be installed.")

        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

        return scaffold


class ButinaSplitter(Splitter):

    def __init__(self, cutoff: float = 0.6):
        """Create a ButinaSplitter.
        Parameters
        ----------
        cutoff: float (default 0.6)
          The cutoff value for tanimoto similarity.  Molecules that are more similar
          than this will tend to be put in the same dataset.
        """
        super().__init__()
        self.cutoff = cutoff

    def split(self,
              dataset: Dataset,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              seed: Optional[int] = None,
              log_every_n: Optional[int] = None,
              homogenous_datasets: bool = True) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits internal compounds into train and validation based on the butina
        clustering algorithm. This splitting algorithm has an O(N^2) run time, where N
        is the number of elements in the dataset. The dataset is expected to be a classification
        dataset.
        This algorithm is designed to generate validation data that are novel chemotypes.
        Setting a small cutoff value will generate smaller, finer clusters of high similarity,
        whereas setting a large cutoff value will generate larger, coarser clusters of low similarity.
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
        log_every_n: int, optional (default None)
          Log every n examples (not currently used).
        homogenous_datasets: bool, optional (default True)
          Whether the datasets will be homogenous or not.
        Returns
        -------
        Tuple[List[int], List[int], List[int]]
          A tuple of train indices, valid indices, and test indices.
        """
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem
            from rdkit.ML.Cluster import Butina
        except ModuleNotFoundError:
            raise ImportError("This function requires RDKit to be installed.")

        mols = generate_mols_and_delete_invalid_smiles(dataset)

        fps_classes_map, indices_classes_map, all_fps = get_fingerprints_for_each_class(mols, dataset)

        is_regression = any([not isinstance(i.item(), int) and not i.item().is_integer() for i in dataset.y])

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


def generate_mols_and_delete_invalid_smiles(dataset: Dataset) -> List[Mol]:
    to_remove = []  # added to deal with invalid smiles and remove them from the dataset #TODO: maybe it would be
    # useful to convert the smiles to Mol upstream in the dataset load
    if dataset.mols.size > 0:
        if any([isinstance(mol, str) for mol in dataset.mols]):
            mols = []

            for i in range(len(dataset.mols)):
                smiles = dataset.mols[i]
                dataset_id = dataset.ids[i]
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if not mol:
                        to_remove.append(dataset_id)
                    else:
                        mols.append(mol)
                except:
                    to_remove.append(dataset_id)

        elif any([isinstance(mol, Mol) for mol in dataset.mols]):
            mols = dataset.mols

        else:
            raise Exception("There are no molecules in the correct format in this dataset")

    else:
        raise Exception("There are no molecules in this dataset")

    if to_remove:
        dataset.remove_elements(to_remove)

    return mols
