import uuid
from abc import abstractmethod, ABC
from typing import Union

import numpy as np
from imblearn import over_sampling, under_sampling, combine
from numpy.random import RandomState
from sklearn.cluster import KMeans

from deepmol.datasets import Dataset
from deepmol.imbalanced_learn._utils import _get_new_ids_smiles_mols


class ImbalancedLearn(ABC):
    """
    Class for dealing with imbalanced datasets.
    A ImbalancedLearn sampler receives a Dataset object and performs over/under sampling.
    Subclasses need to implement a _sample method to perform over/under sampling.
    """

    def __init__(self) -> None:
        """
        Initialize the ImbalancedLearn sampler.
        """
        if self.__class__ == ImbalancedLearn:
            raise Exception('Abstract class ImbalancedLearn should not be instantiated')

        self.features = None
        self.y = None
        self.ids = None
        self.smiles = None
        self.mols = None

    def sample(self, dataset: Dataset) -> Dataset:
        """
        Sample the dataset according to the sampling strategy.
        Parameters
        ----------
        dataset: Dataset
            Dataset to sample.
        Returns
        -------
        dataset: Dataset
            Sampled dataset.
        """
        self.features = dataset.X
        self.y = dataset.y
        self.ids = dataset.ids
        self.smiles = dataset.smiles
        self.mols = dataset.mols
        features, y, ids, smiles, mols = self._sample()
        dataset._X = features
        dataset._y = y
        dataset._ids = ids
        dataset._smiles = smiles
        dataset._mols = mols
        return dataset

    @abstractmethod
    def _sample(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Perform over/under sampling.
        Returns
        -------
        features: np.ndarray
            Features of the sampled dataset.
        y: np.ndarray
            Labels of the sampled dataset.
        ids: np.ndarray
            Ids of the sampled dataset.
        smiles: np.ndarray
            Smiles of the sampled dataset.
        mols: np.ndarray
            Mols of the sampled dataset.
        """


#########################################
# OVER-SAMPLING
#########################################

class RandomOverSampler(ImbalancedLearn):
    """
    Class to perform naive random over-sampling.
    Wrapper around ImbalancedLearn RandomOverSampler
    (https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.RandomOverSampler.html)
    Object to over-sample the minority class(es) by picking samples at random with replacement.
    """

    def __init__(self,
                 sampling_strategy: Union[float, str, dict, callable] = "auto",
                 random_state: Union[int, RandomState] = None) -> None:
        """
        Initialize the RandomOverSampler.
        Parameters
        ----------
        sampling_strategy: Union[float, str, dict, callable]
            Sampling information to resample the data set.
            When float, it corresponds to the desired ratio of the number of samples in the minority class over
            the number of samples in the majority class after resampling.
            When str, specify the class targeted by the resampling. The number of samples in the different classes
            will be equalized. Possible choices are:
                'minority': resample only the minority class;
                'not minority': resample all classes but the minority class;
                'not majority': resample all classes but the majority class;
                'all': resample all classes;
                'auto': equivalent to 'not majority'.
            When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
            samples for each targeted class.
            When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
            The values correspond to the desired number of samples for each class.
        random_state: Union[int, RandomState]
            Control the randomization of the algorithm.
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.
        """
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    @staticmethod
    def _get_new_ids(ids, idx):
        """
        Returns new ids for the resampled dataset. If the sample is artificial, the id is prefixed with "ib_", followed
        by the original id.
        Parameters
        ----------
        ids: np.ndarray
            Original ids.
        idx: np.ndarray
            Indexes of the resampled dataset.
        Returns
        -------
        new_ids: np.ndarray
            New ids for the resampled dataset.
        """
        new_ids = []
        seen_indexes = {}
        for i in idx:
            if i not in seen_indexes:
                new_ids.append(ids[i])
                seen_indexes[i] = 1
            else:
                if f"ib{seen_indexes[i] - 1}_{ids[i]}" not in new_ids:
                    new_ids.append(f"ib{seen_indexes[i] - 1}_{ids[i]}")
                seen_indexes[i] += 1
        return new_ids

    def _sample(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Compute the over-sampling.
        Returns
        -------
        x: np.ndarray
            Features of the sampled dataset.
        y: np.ndarray
            Labels of the sampled dataset.
        ids: np.ndarray
            Ids of the sampled dataset.
        smiles: np.ndarray
            Smiles of the sampled dataset.
        mols: np.ndarray
            Mols of the sampled dataset.
        """
        ros = over_sampling.RandomOverSampler(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
        x, y = ros.fit_resample(self.features, self.y)
        indexes = ros.sample_indices_
        ids = self._get_new_ids(self.ids, indexes)
        smiles = self.smiles[indexes]
        mols = self.mols[indexes]
        return x, y, ids, smiles, mols


class SMOTE(ImbalancedLearn):
    """
    Class to perform Synthetic Minority Oversampling Technique (SMOTE) over-sampling.
    Wrapper around ImbalancedLearn SMOTE
    (https://imbalanced-learn.org/stable/generated/imblearn.over_sampling.SMOTE.html)
    """

    def __init__(self,
                 sampling_strategy: Union[float, str, dict, callable] = "auto",
                 random_state: Union[int, RandomState] = None,
                 k_neighbors: int = 5,
                 n_jobs: int = None) -> None:
        """
        Initialize the SMOTE.
        Parameters
        ----------
        sampling_strategy: Union[float, str, dict, callable]
            Sampling information to resample the data set.
            When float, it corresponds to the desired ratio of the number of samples in the minority class over
            the number of samples in the majority class after resampling.
            When str, specify the class targeted by the resampling. The number of samples in the different classes
            will be equalized. Possible choices are:
                'minority': resample only the minority class;
                'not minority': resample all classes but the minority class;
                'not majority': resample all classes but the majority class;
                'all': resample all classes;
                'auto': equivalent to 'not majority'.
            When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
            samples for each targeted class.
            When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
            The values correspond to the desired number of samples for each class.
        random_state: Union[int, RandomState]
            Control the randomization of the algorithm.
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.
        k_neighbors: int
            Number of nearest neighbours to used to construct synthetic samples.
        n_jobs: int
            Number of CPU cores used during the cross-validation loop. None means 1 unless in a
            joblib.parallel_backend context. -1 means using all processors.
        """
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def _sample(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Compute the over-sampling.
        Returns
        -------
        x: np.ndarray
            Features of the sampled dataset.
        y: np.ndarray
            Labels of the sampled dataset.
        ids: np.ndarray
            Ids of the sampled dataset.
        smiles: np.ndarray
            Smiles of the sampled dataset.
        mols: np.ndarray
            Mols of the sampled dataset.
        """
        ros = over_sampling.SMOTE(sampling_strategy=self.sampling_strategy,
                                  random_state=self.random_state,
                                  k_neighbors=self.k_neighbors)
        x, y = ros.fit_resample(self.features, self.y)
        # list of original ids + artificial ids
        ids = np.concatenate((self.ids, [f"ib_{uuid.uuid4().hex}" for _ in range(x.shape[0] - self.features.shape[0])]))
        smiles = np.concatenate((self.smiles, [None for _ in range(x.shape[0] - self.features.shape[0])]))
        mols = np.concatenate((self.mols, [None for _ in range(x.shape[0] - self.features.shape[0])]))
        return x, y, ids, smiles, mols


#########################################
# UNDER-SAMPLING
#########################################

class ClusterCentroids(ImbalancedLearn):
    """
    Class to perform ClusterCentroids under-sampling.
    Wrapper around ImbalancedLearn ClusterCentroids
    (https://imbalanced-learn.org/stable/generated/imblearn.under_sampling.ClusterCentroids.html)
    Perform under-sampling by generating centroids based on clustering.
    """

    def __init__(self,
                 sampling_strategy: Union[float, str, dict, callable] = "auto",
                 random_state: Union[int, RandomState] = None,
                 estimator: callable = KMeans(),
                 voting: str = 'auto') -> None:
        """
        Initialize the ClusterCentroids.
        Parameters
        ----------
        sampling_strategy: Union[float, str, dict, callable]
            Sampling information to resample the data set.
            When float, it corresponds to the desired ratio of the number of samples in the minority class over
            the number of samples in the majority class after resampling.
            When str, specify the class targeted by the resampling. The number of samples in the different classes
            will be equalized. Possible choices are:
                'minority': resample only the minority class;
                'not minority': resample all classes but the minority class;
                'not majority': resample all classes but the majority class;
                'all': resample all classes;
                'auto': equivalent to 'not majority'.
            When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
            samples for each targeted class.
            When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
            The values correspond to the desired number of samples for each class.
        random_state: Union[int, RandomState]
            Control the randomization of the algorithm.
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.
        estimator: object, default=KMeans()
            Pass a sklearn.cluster.KMeans estimator.
        voting: str
            Voting strategy to generate the new samples:
            If 'hard', the nearest-neighbors of the centroids found using the clustering algorithm will be used.
            If 'soft', the centroids found by the clustering algorithm will be used.
            If 'auto', if the input is sparse, it will default on 'hard' otherwise, 'soft' will be used.
        """
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.estimator = estimator
        self.voting = voting

    def _sample(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Compute the under-sampling.
        Returns
        -------
        x: np.ndarray
            Features of the sampled dataset.
        y: np.ndarray
            Labels of the sampled dataset.
        ids: np.ndarray
            Ids of the sampled dataset.
        smiles: np.ndarray
            Smiles of the sampled dataset.
        mols: np.ndarray
            Mols of the sampled dataset.
        """
        ros = under_sampling.ClusterCentroids(sampling_strategy=self.sampling_strategy,
                                              random_state=self.random_state,
                                              estimator=self.estimator,
                                              voting=self.voting)
        x, y = ros.fit_resample(self.features, self.y)
        ids, smiles, mols = _get_new_ids_smiles_mols(self.features, x, self.ids, self.smiles, self.mols)
        return x, y, ids, smiles, mols


class RandomUnderSampler(ImbalancedLearn):
    """
    Class to perform RandomUnderSampler under-sampling.
    Wrapper around ImbalancedLearn RandomUnderSampler
    (https://imbalanced-learn.org/stable/generated/imblearn.under_sampling.RandomUnderSampler.html)
    Under-sample the majority class(es) by randomly picking samples with or without replacement.
    """

    def __init__(self,
                 sampling_strategy: Union[float, str, dict, callable] = "auto",
                 random_state: Union[int, RandomState] = None,
                 replacement: bool = False) -> None:
        """
        Initialize the RandomUnderSampler.
        Parameters
        ----------
        sampling_strategy: Union[float, str, dict, callable]
            Sampling information to resample the data set.
            When float, it corresponds to the desired ratio of the number of samples in the minority class over
            the number of samples in the majority class after resampling.
            When str, specify the class targeted by the resampling. The number of samples in the different classes
            will be equalized. Possible choices are:
                'minority': resample only the minority class;
                'not minority': resample all classes but the minority class;
                'not majority': resample all classes but the majority class;
                'all': resample all classes;
                'auto': equivalent to 'not majority'.
            When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
            samples for each targeted class.
            When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
            The values correspond to the desired number of samples for each class.
        random_state: Union[int, RandomState]
            Control the randomization of the algorithm.
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.
        replacement: bool
            Whether the sample is with or without replacement.
        """
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.replacement = replacement

    def _sample(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Compute the under-sampling.
        Returns
        -------
        x: np.ndarray
            Features of the sampled dataset.
        y: np.ndarray
            Labels of the sampled dataset.
        ids: np.ndarray
            Ids of the sampled dataset.
        smiles: np.ndarray
            Smiles of the sampled dataset.
        mols: np.ndarray
            Mols of the sampled dataset.
        """
        ros = under_sampling.RandomUnderSampler(sampling_strategy=self.sampling_strategy,
                                                random_state=self.random_state,
                                                replacement=self.replacement)
        x, y = ros.fit_resample(self.features, self.y)
        indexes = ros.sample_indices_
        ids = self.ids[indexes]
        smiles = self.smiles[indexes]
        mols = self.mols[indexes]
        return x, y, ids, smiles, mols


#########################################
# COMBINATION OF OVER AND UNDER-SAMPLING
#########################################

class SMOTEENN(ImbalancedLearn):
    """
    Class to perform SMOTEENN over and under-sampling.
    Wrapper around ImbalancedLearn SMOTEENN
    (https://imbalanced-learn.org/stable/generated/imblearn.combine.SMOTEENN.html)
    Over-sampling using SMOTE and cleaning using ENN.
    Combine over and under-sampling using SMOTE and Edited Nearest Neighbours.
    """

    def __init__(self,
                 sampling_strategy: Union[float, str, dict, callable] = "auto",
                 random_state: Union[int, RandomState] = None,
                 smote: callable = None,
                 enn: callable = None,
                 n_jobs: int = None) -> None:
        """
        Initialize the SMOTEENN.
        Parameters
        ----------
        sampling_strategy: Union[float, str, dict, callable]
            Sampling information to resample the data set.
            When float, it corresponds to the desired ratio of the number of samples in the minority class over
            the number of samples in the majority class after resampling.
            When str, specify the class targeted by the resampling. The number of samples in the different classes
            will be equalized. Possible choices are:
                'minority': resample only the minority class;
                'not minority': resample all classes but the minority class;
                'not majority': resample all classes but the majority class;
                'all': resample all classes;
                'auto': equivalent to 'not majority'.
            When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
            samples for each targeted class.
            When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
            The values correspond to the desired number of samples for each class.
        random_state: Union[int, RandomState]
            Control the randomization of the algorithm.
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.
        smote: callable
            The imblearn.over_sampling.SMOTE object to use. If not given, a imblearn.over_sampling.SMOTE object
            with default parameters will be given.
        enn: callable
            The imblearn.under_sampling.EditedNearestNeighbours object to use. If not given, a
            imblearn.under_sampling.EditedNearestNeighbours object with sampling strategy=’all’ will be given.
        n_jobs: int
            Number of CPU cores used during the cross-validation loop. None means 1 unless in a
            joblib.parallel_backend context. -1 means using all processors.
        """
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = smote
        self.enn = enn
        self.n_jobs = n_jobs

    def _sample(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Compute the under-sampling and over-sampling.
        Returns
        -------
        x: np.ndarray
            Features of the sampled dataset.
        y: np.ndarray
            Labels of the sampled dataset.
        ids: np.ndarray
            Ids of the sampled dataset.
        smiles: np.ndarray
            Smiles of the sampled dataset.
        mols: np.ndarray
            Mols of the sampled dataset.
        """
        ros = combine.SMOTEENN(sampling_strategy=self.sampling_strategy,
                               random_state=self.random_state,
                               smote=self.smote,
                               enn=self.enn,
                               n_jobs=self.n_jobs)
        x, y = ros.fit_resample(self.features, self.y)
        ids, smiles, mols = _get_new_ids_smiles_mols(self.features, x, self.ids, self.smiles, self.mols)
        return x, y, ids, smiles, mols


class SMOTETomek(ImbalancedLearn):
    """
    Class to perform SMOTETomek over and under-sampling.
    Wrapper around ImbalancedLearn SMOTETomek
    (https://imbalanced-learn.org/stable/generated/imblearn.combine.SMOTETomek.html)
    Over-sampling using SMOTE and cleaning using Tomek links.
    Combine over- and under-sampling using SMOTE and Tomek links.
    """

    def __init__(self,
                 sampling_strategy: Union[float, str, dict, callable] = "auto",
                 random_state: Union[int, RandomState] = None,
                 smote: callable = None,
                 tomek: callable = None,
                 n_jobs: int = None) -> None:
        """
        Initialize the SMOTETomek.
        Parameters
        ----------
        sampling_strategy: Union[float, str, dict, callable]
            Sampling information to resample the data set.
            When float, it corresponds to the desired ratio of the number of samples in the minority class over
            the number of samples in the majority class after resampling.
            When str, specify the class targeted by the resampling. The number of samples in the different classes
            will be equalized. Possible choices are:
                'minority': resample only the minority class;
                'not minority': resample all classes but the minority class;
                'not majority': resample all classes but the majority class;
                'all': resample all classes;
                'auto': equivalent to 'not majority'.
            When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
            samples for each targeted class.
            When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
            The values correspond to the desired number of samples for each class.
        random_state: Union[int, RandomState]
            Control the randomization of the algorithm.
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.
        smote: callable
            The imblearn.over_sampling.SMOTE object to use. If not given, a imblearn.over_sampling.SMOTE object
            with default parameters will be given.
        tomek: callable
            The imblearn.under_sampling.TomekLinks object to use. If not given, a imblearn.under_sampling.TomekLinks
            object with sampling strategy="all" will be given.
        n_jobs: int
            Number of CPU cores used during the cross-validation loop. None means 1 unless in a
            joblib.parallel_backend context. -1 means using all processors.
        """
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = smote
        self.tomek = tomek
        self.n_jobs = n_jobs

    def _sample(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Compute the under-sampling and over-sampling.
        Returns
        -------
        x: np.ndarray
            Features of the sampled dataset.
        y: np.ndarray
            Labels of the sampled dataset.
        ids: np.ndarray
            Ids of the sampled dataset.
        smiles: np.ndarray
            Smiles of the sampled dataset.
        mols: np.ndarray
            Mols of the sampled dataset.
        """
        ros = combine.SMOTETomek(sampling_strategy=self.sampling_strategy,
                                 random_state=self.random_state,
                                 smote=self.smote,
                                 tomek=self.tomek,
                                 n_jobs=self.n_jobs)
        x, y = ros.fit_resample(self.features, self.y)
        ids, smiles, mols = _get_new_ids_smiles_mols(self.features, x, self.ids, self.smiles, self.mols)
        return x, y, ids, smiles, mols
