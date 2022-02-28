from abc import ABC, abstractmethod

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import Mol

from datasets.datasets import Dataset
from scalers.base_scaler import BaseScaler
from utils.errors import PreConditionViolationException


class MolecularFeaturizer(ABC):
    """Abstract class for calculating a set of features for a molecule.
    A `MolecularFeaturizer` uses SMILES strings or RDKit molecule
    objects to represent molecules.

    Subclasses need to implement the _featurize method for
    calculating features for a single molecule.
    """

    def __init__(self, ):

        if self.__class__ == MolecularFeaturizer:
            raise Exception('Abstract class MolecularFeaturizer should not be instantiated')

    def featurize(self, dataset: Dataset, log_every_n=1000,
                  scaler: BaseScaler = None,
                  path_to_save_scaler: str = None,
                  remove_nans_axis: int = 0):

        """Calculate features for molecules.
        Parameters
        ----------
        dataset: Dataset object
          Dataset containing molecules to featurize
        log_every_n: int, default 1000
          Logging messages reported every `log_every_n` samples.
        scaler: BaseScaler, default None
          Scale the data
        path_to_save_scaler: str, default None
          File path to save scaler
        remove_nans_axis: int, default 0
          integer that defines in which axis the NaNs should be removed
        Returns
        -------
        dataset: Dataset object
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
                    molobj = Chem.MolFromSmiles(mol)
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
                if isinstance(mol, Chem.rdchem.Mol):
                    mol = Chem.MolToSmiles(mol)
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
