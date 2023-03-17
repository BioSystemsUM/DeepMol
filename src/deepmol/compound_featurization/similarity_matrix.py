from typing import Tuple

import numpy as np

from deepmol.compound_featurization._utils import calc_morgan_fingerprints, calc_similarity
from deepmol.datasets import Dataset
from deepmol.parallelism.multiprocessing import JoblibMultiprocessing


class TanimotoSimilarityMatrix:

    def __init__(self, n_molecules: int, n_jobs: int = -1) -> None:
        """
        Initialize a MACCSkeysFingerprint object.

        Parameters
        ----------
        n_molecules: int
            Number of molecules in the dataset.
        n_jobs: int
            Number of jobs to run in parallel.
        """
        self.n_jobs = n_jobs
        self.feature_names = [f"tanimoto_{i}" for i in range(n_molecules)]
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
        n_mols = len(dataset.smiles)
        similarity_matrix = np.zeros((n_mols, n_mols), dtype=np.float32)

        # use multiprocessing to calculate similarities in parallel
        multiprocessing_cls = JoblibMultiprocessing(process=self._calc_similarity, n_jobs=self.n_jobs)
        pairs = [(i, j) for i in range(n_mols) for j in range(i + 1, n_mols)]
        features = multiprocessing_cls.run(pairs)
        for i, j, similarity in features:
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

        features = np.array(similarity_matrix, dtype=object)

        dataset._X = features
        dataset.feature_names = self.feature_names
        return dataset
