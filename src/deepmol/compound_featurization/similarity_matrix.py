from typing import Tuple

import numpy as np

from deepmol.base import Transformer
from deepmol.compound_featurization._utils import calc_morgan_fingerprints, calc_similarity
from deepmol.datasets import Dataset
from deepmol.parallelism.multiprocessing import JoblibMultiprocessing
from deepmol.utils.decorators import modify_object_inplace_decorator


class TanimotoSimilarityMatrix(Transformer):
    """
    Class to calculate Tanimoto similarity matrix for a dataset.

    The similarity matrix is calculated using Morgan fingerprints.
    """

    def __init__(self, n_molecules: int = None, n_jobs: int = -1) -> None:
        """
        Initialize a TanimotoSimilarityMatrix object.

        Parameters
        ----------
        n_molecules: int
            Number of molecules to evaluate the similarity with. It choses the first n_molecules from the dataset.
        n_jobs: int
            Number of jobs to run in parallel.
        """
        super().__init__()
        self.n_molecules = n_molecules
        self.n_jobs = n_jobs
        self.feature_names = [f"tanimoto_{i}" for i in range(self.n_molecules)]
        self.fps = None

    def _calc_similarity(self, i, j) -> Tuple[int, int, float]:
        """
        Calculates the Tanimoto similarity between two fingerprints.

        Parameters
        ----------
        i: int
            Index of the first fingerprint
        j: int
            Index of the second fingerprint

        Returns
        -------
        i: int
            Index of the first fingerprint
        j: int
            Index of the second fingerprint
        similarity: float
            Tanimoto similarity between the two fingerprints
        """
        return calc_similarity(i, j, self.fps)

    @modify_object_inplace_decorator
    def featurize(self,
                  dataset: Dataset,
                  **kwargs
                  ) -> Dataset:
        """
        Calculate Tanimoto similarities between all pairs of molecules in the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to featurize.
        **kwargs
            Keyword arguments to pass to the Morgan fingerprint function.

        Returns
        -------
        dataset: Dataset
            The dataset with the Tanimoto similarities as features.
        """
        self.fps = calc_morgan_fingerprints(dataset.mols, **kwargs)
        # initialize similarity matrix as a numpy array
        if self.n_molecules is None:
            self.n_molecules = len(dataset.mols)
        n_mols = len(dataset.smiles)
        similarity_matrix = np.zeros((n_mols, self.n_molecules), dtype=np.float32)

        # use multiprocessing to calculate similarities in parallel
        multiprocessing_cls = JoblibMultiprocessing(process=self._calc_similarity, n_jobs=self.n_jobs)
        pairs = [(i, j) for i in range(n_mols) for j in range(i + 1, self.n_molecules)]
        features = multiprocessing_cls.run(pairs)
        for i, j, similarity in features:
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

        features = np.array(similarity_matrix, dtype=object)

        dataset._X = features
        dataset.feature_names = self.feature_names
        return dataset

    def _fit(self, dataset: Dataset) -> 'TanimotoSimilarityMatrix':
        """
        Fit the featurizer to a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the featurizer to.

        Returns
        -------
        self: Mol2Vec
            The fitted featurizer.
        """
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.
        """
        return self.featurize(dataset)
