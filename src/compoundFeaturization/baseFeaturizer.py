import numpy as np
from typing import Iterable, Any

from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import Mol

from Datasets.Datasets import Dataset


class MolecularFeaturizer(object):
    """Abstract class for calculating a set of features for a molecule.
    A `MolecularFeaturizer` uses SMILES strings or RDKit molecule
    objects to represent molecules.

    Subclasses need to implement the _featurize method for
    calculating features for a single molecule.
    """

    def __init__(self):
        if self.__class__ == MolecularFeaturizer:
            raise Exception('Abstract class MolecularFeaturizer should not be instantiated')

    def featurize(self, dataset: Dataset, log_every_n=1000):
        """Calculate features for molecules.
        Parameters
        ----------
        dataset: Dataset object
          Dataset containing molecules to featurize
        log_every_n: int, default 1000
          Logging messages reported every `log_every_n` samples.
        Returns
        -------
        dataset: Dataset object
          The input Dataset containing a featurized representation of the molecules in Dataset.X.
        """
        molecules = dataset.mols

        features = []
        for i, mol in enumerate(molecules):
            if i % log_every_n == 0:
                print("Featurizing datapoint %i" % i)
            try:
                if isinstance(mol, str):
                    # mol must be a RDKit Mol object, so parse a SMILES
                    molobj = Chem.MolFromSmiles(mol)
                    try :
                        # SMILES is unique, so set a canonical order of atoms
                        new_order = rdmolfiles.CanonicalRankAtoms(molobj)
                        molobj = rdmolops.RenumberAtoms(molobj, new_order)
                        mol = molobj
                    except Exception as e:
                        mol = mol
                features.append(self._featurize(mol))
            except Exception as e:
                if isinstance(mol, Chem.rdchem.Mol):
                    mol = Chem.MolToSmiles(mol)
                print("Failed to featurize datapoint %d, %s. Appending empty array" %(i, mol))
                print("Exception message: {}".format(e))
        dataset.X = np.asarray(features)

        #if anyNA:
        #TODO: where and in which manner to remove NAs????
        dataset.removeNAs()

        return dataset

    # TODO: implement
    def load_fp(self):
        pass