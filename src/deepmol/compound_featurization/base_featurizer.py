import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
from rdkit.Chem import MolFromSmiles, rdmolfiles, rdmolops, Mol, MolToSmiles

from deepmol.datasets import Dataset
from deepmol.loggers.logger import Logger
from deepmol.parallelism.multiprocessing import JoblibMultiprocessing
from deepmol.scalers import BaseScaler
from deepmol.utils.errors import PreConditionViolationException
from deepmol.utils.utils import canonicalize_mol_object


class MolecularFeaturizer(ABC):
    """
    Abstract class for calculating a set of features for a molecule.
    A `MolecularFeaturizer` uses SMILES strings or RDKit molecule objects to represent molecules.

    Subclasses need to implement the _featurize method for calculating features for a single molecule.
    """

    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs

        self.logger = Logger()

    @staticmethod
    def _convert_smiles_to_mol(mol: str) -> Tuple[Mol, bool, bool]:
        """
        Convert a SMILES string to a RDKit molecule object.

        Parameters
        ----------
        mol: str
            The SMILES string to convert.

        Returns
        -------
        mol: Mol
            The RDKit molecule object.
        is_mol_convertable: bool
            Whether the SMILES string could be converted to a RDKit molecule object.
        remove_mol: bool
            Whether the molecule should be removed from the dataset.
        """

        is_mol_convertable = True
        remove_mol = False

        mol_object = MolFromSmiles(mol)
        if mol_object is None:
            remove_mol = True
            is_mol_convertable = False

        mol = canonicalize_mol_object(mol_object)

        return mol, is_mol_convertable, remove_mol

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
        """
        is_mol_convertable = True
        remove_mol = False
        smiles = None
        try:
            if isinstance(mol, str):
                # mol must be a RDKit Mol object, so parse a SMILES
                smiles = mol
                mol, is_mol_convertable, remove_mol = self._convert_smiles_to_mol(mol)
            elif isinstance(mol, Mol):
                mol = canonicalize_mol_object(mol)
            else:
                is_mol_convertable = False
                remove_mol = True

            if is_mol_convertable:
                feat = self._featurize(mol)
                return feat, remove_mol
            else:
                if isinstance(mol, Mol):
                    smiles = MolToSmiles(mol)
                logger = logging.getLogger(self.logger.logger)
                logger.error(f"Failed to featurize {smiles}. Appending empty array")
                return np.array([]), remove_mol

        except PreConditionViolationException:
            exit(1)

        except Exception as e:
            logger = logging.getLogger(self.logger.logger)
            logger.error("Exception message: {}".format(e))
            remove_mol = True
            return np.array([]), remove_mol

    def featurize(self,
                  dataset: Dataset,
                  scaler: BaseScaler = None,
                  path_to_save_scaler: str = None,
                  remove_nans_axis: int = 0
                  ):

        """
        Calculate features for molecules.

        Parameters
        ----------
        dataset: Dataset
            The dataset containing the molecules to featurize in dataset.mols.
        scaler: BaseScaler
            The scaler to use for scaling the generated features.
        path_to_save_scaler: str
            The path to save the scaler to.
        remove_nans_axis: int
            The axis to remove NaNs from. If None, no NaNs are removed.

        Returns
        -------
        dataset: Dataset
          The input Dataset containing a featurized representation of the molecules in Dataset.X.
        """
        molecules = dataset.mols
        # dataset_ids = dataset.ids

        multiprocessing_cls = JoblibMultiprocessing(process=self._featurize_mol, n_jobs=self.n_jobs)
        features = multiprocessing_cls.run(molecules)

        features, remove_mols = zip(*features)

        remove_mols_list = np.array(remove_mols)
        dataset.remove_elements(dataset.ids[remove_mols_list])

        features = np.array(features, dtype=object)
        features = features[~remove_mols_list]
        # features = np.concatenate(features, axis=0)

        if isinstance(features[0], np.ndarray):
            features = np.vstack(features)
        dataset.X = features

        dataset.remove_nan(remove_nans_axis)

        if scaler and path_to_save_scaler:
            # transform data
            scaler.fit_transform(dataset)
            scaler.save(path_to_save_scaler)

        elif scaler:
            scaler.transform(dataset)

        return dataset

    @abstractmethod
    def _featurize(self, mol: Mol):
        raise NotImplementedError
