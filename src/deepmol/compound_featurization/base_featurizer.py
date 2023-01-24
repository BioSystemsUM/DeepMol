from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
from rdkit.Chem import MolFromSmiles, rdmolfiles, rdmolops, Mol, MolToSmiles

from deepmol.datasets import Dataset
from deepmol.parallelism.multiprocessing import JoblibMultiprocessing
from deepmol.scalers import BaseScaler
from deepmol.utils.errors import PreConditionViolationException


class MolecularFeaturizer(ABC):
    """
    Abstract class for calculating a set of features for a molecule.
    A `MolecularFeaturizer` uses SMILES strings or RDKit molecule objects to represent molecules.

    Subclasses need to implement the _featurize method for calculating features for a single molecule.
    """

    def __init__(self):
        if self.__class__ == MolecularFeaturizer:
            raise Exception('Abstract class MolecularFeaturizer should not be instantiated')

        self.multiprocessing_cls = JoblibMultiprocessing()

    def _featurize_mol(self, mol: Mol, mol_id: Union[int, str]) -> Tuple[np.ndarray, bool]:
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
        mol_convertable = True

        remove_mol = False

        try:
            if isinstance(mol, str):
                # mol must be a RDKit Mol object, so parse a SMILES
                mol_object = MolFromSmiles(mol)
                if mol_object is None:
                    remove_mol = True
                    mol_convertable = False
                try:
                    # SMILES is unique, so set a canonical order of atoms
                    new_order = rdmolfiles.CanonicalRankAtoms(mol_object)
                    mol_object = rdmolops.RenumberAtoms(mol_object, new_order)
                    mol = mol_object
                except Exception as e:
                    mol = mol

            if mol_convertable:
                feat = self._featurize(mol)
                return feat

        except PreConditionViolationException:
            exit(1)

        except Exception as e:
            if isinstance(mol, Mol):
                mol = MolToSmiles(mol)
            print("Failed to featurize datapoint %d, %s. Appending empty array" % (mol_id, mol))
            print("Exception message: {}".format(e))
            remove_mol = True

        return feat, remove_mol

    def featurize(self,
                  dataset: Dataset,
                  scaler: BaseScaler = None,
                  path_to_save_scaler: str = None,
                  remove_nans_axis: int = 0,
                  log_every_n: int = 1000):

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
        log_every_n: int
            Logging messages reported every `log_every_n` samples.

        Returns
        -------
        dataset: Dataset
          The input Dataset containing a featurized representation of the molecules in Dataset.X.
        """
        molecules = dataset.mols
        dataset_ids = dataset.ids

        features = []

        for i, mol in enumerate(molecules):
            self._featurize_mol(mol, dataset_ids[i])

        if isinstance(features[0], np.ndarray):
            features = np.vstack(features)
        dataset.X = features

        dataset.remove_nan(remove_nans_axis)

        if scaler and path_to_save_scaler:
            # transform data
            scaler.fit_transform(dataset)
            scaler.save_scaler(path_to_save_scaler)

        elif scaler:
            scaler.transform(dataset)

        return dataset

    @abstractmethod
    def _featurize(self, mol: Mol):
        raise NotImplementedError()
