from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from rdkit.Chem import Mol, MolToSmiles

from deepmol.base import Transformer
from deepmol.datasets import Dataset
from deepmol.loggers.logger import Logger
from deepmol.parallelism.multiprocessing import JoblibMultiprocessing
from deepmol.utils.decorators import modify_object_inplace_decorator
from deepmol.utils.errors import PreConditionViolationException
from deepmol.utils.utils import canonicalize_mol_object


class MolecularFeaturizer(ABC, Transformer):
    """
    Abstract class for calculating a set of features for a molecule.
    A `MolecularFeaturizer` uses SMILES strings or RDKit molecule objects to represent molecules.

    Subclasses need to implement the _featurize method for calculating features for a single molecule.
    """

    def __init__(self, n_jobs: int = -1) -> None:
        """
        Initializes the featurizer.

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel in the featurization.
        """
        super().__init__()
        self.n_jobs = n_jobs
        self.feature_names = None
        self.logger = Logger()

    def _featurize_mol(self, mol: Mol) -> Tuple[np.ndarray, bool]:
        """
        Calculate features for a single molecule.

        Parameters
        ----------
        mol: Mol
            The molecule to featurize.

        Returns
        -------
        features: np.ndarray
            The features for the molecule.
        remove_mol: bool
            Whether the molecule should be removed from the dataset.
        """
        try:
            mol = canonicalize_mol_object(mol)
            feat = self._featurize(mol)
            remove_mol = False
            return feat, remove_mol
        except PreConditionViolationException:
            exit(1)

        except Exception as e:
            if mol is not None:
                smiles = MolToSmiles(mol)
            else:
                smiles = None
            self.logger.error(f"Failed to featurize {smiles}. Appending empty array")
            self.logger.error("Exception message: {}".format(e))
            remove_mol = True
            return np.array([]), remove_mol

    @modify_object_inplace_decorator
    def featurize(self,
                  dataset: Dataset,
                  remove_nans_axis: int = 0
                  ) -> Dataset:

        """
        Calculate features for molecules.

        Parameters
        ----------
        dataset: Dataset
            The dataset containing the molecules to featurize in dataset.mols.
        remove_nans_axis: int
            The axis to remove NaNs from. If None, no NaNs are removed.

        Returns
        -------
        dataset: Dataset
          The input Dataset containing a featurized representation of the molecules in Dataset.X.
        """
        molecules = dataset.mols
        multiprocessing_cls = JoblibMultiprocessing(process=self._featurize_mol, n_jobs=self.n_jobs)
        features = multiprocessing_cls.run(molecules)

        features, remove_mols = zip(*features)

        remove_mols_list = np.array(remove_mols)
        dataset.remove_elements(np.array(dataset.ids)[remove_mols_list], inplace=True)
        
        features = np.array(features, dtype=object)
        features = features[~remove_mols_list]
        
        try:
            features = features.astype('float64')
        except:
            pass

        if (isinstance(features[0], np.ndarray) and len(features[0].shape) == 2) or not isinstance(features[0],
                                                                                                   np.ndarray):
            pass
        else:
            features = np.vstack(features)

        dataset.clear_cached_properties()
        dataset._X = features
        dataset.feature_names = self.feature_names
        dataset.remove_nan(remove_nans_axis, inplace=True)
        return dataset

    @abstractmethod
    def _featurize(self, mol: Mol):
        raise NotImplementedError

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Calculate features for molecules. This method is called by the transform method of the Transformer class.
        To be used by pipeline.

        Parameters
        ----------
        dataset: Dataset
            The dataset containing the molecules to featurize in dataset.mols.

        Returns
        -------
        dataset: Dataset
            The input Dataset containing a featurized representation of the molecules in Dataset.X.
        """
        return self.featurize(dataset)

    def _fit(self, dataset: Dataset) -> 'MolecularFeaturizer':
        """
        Fit the featurizer. This method is called by the fit method of the Transformer class.
        To be used by pipeline.

        Parameters
        ----------
        dataset: Dataset
            The dataset containing the molecules to featurize in dataset.mols.

        Returns
        -------
        self: MolecularFeaturizer
            The fitted featurizer.
        """
        return self
