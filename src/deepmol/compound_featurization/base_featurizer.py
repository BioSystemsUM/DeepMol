from abc import ABC, abstractmethod

import numpy as np
from rdkit.Chem import MolFromSmiles, rdmolfiles, rdmolops, Mol, MolToSmiles

from deepmol.datasets import Dataset
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
            mol_id = dataset_ids[i]
            mol_convertable = True
            if i % log_every_n == 0:
                print("Featurizing datapoint %i" % i)
            try:
                if isinstance(mol, str):
                    # mol must be a RDKit Mol object, so parse a SMILES
                    molobj = MolFromSmiles(mol)
                    if molobj is None:
                        dataset.remove_elements([mol_id])
                        mol_convertable = False
                    try:
                        # SMILES is unique, so set a canonical order of atoms
                        new_order = rdmolfiles.CanonicalRankAtoms(molobj)
                        molobj = rdmolops.RenumberAtoms(molobj, new_order)
                        mol = molobj
                    except Exception as e:
                        mol = mol

                if mol_convertable:
                    feat = self._featurize(mol)
                    features.append(feat)

            except PreConditionViolationException:
                exit(1)

            except Exception as e:
                if isinstance(mol, Mol):
                    mol = MolToSmiles(mol)
                print("Failed to featurize datapoint %d, %s. Appending empty array" % (i, mol))
                print("Exception message: {}".format(e))
                dataset.remove_elements([mol_id])

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
