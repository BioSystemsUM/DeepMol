from compoundFeaturization.baseFeaturizer import MolecularFeaturizer
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols
import numpy as np
from typing import Any

class MorganFingerprint(MolecularFeaturizer):
    """Morgan fingerprints.
    Extended Connectivity Circular Fingerprints compute a bag-of-words style
    representation of a molecule by breaking it into local neighborhoods and
    hashing into a bit vector of the specified size.
    """

    def __init__(self,
                 radius: int = 2,
                 size: int = 1024,
                 chiral: bool = False,
                 bonds: bool = True,
                 features: bool = False):
        """
        Parameters
        ----------
        radius: int, optional (default 2)
          Fingerprint radius.
        size: int, optional (default 1024)
          Length of generated bit vector.
        chiral: bool, optional (default False)
          Whether to consider chirality in fingerprint generation.
        bonds: bool, optional (default True)
          Whether to consider bond order in fingerprint generation.
        features: bool, optional (default False)
          Whether to use feature information instead of atom information;
        """

        self.radius = radius
        self.size = size
        self.chiral = chiral
        self.bonds = bonds
        self.features = features

    def _featurize(self, mol: Any) -> np.ndarray:
        """Calculate morgan fingerprint.
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of circular fingerprint.
        """

        try :
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
                                                                self.radius,
                                                                nBits=self.size,
                                                                useChirality=self.chiral,
                                                                useBondTypes=self.bonds,
                                                                useFeatures=self.features)
        except Exception as e:
            #print(e)
            print('error in smile: ' + str(mol))
            fp = np.nan
        fp = np.asarray(fp, dtype=np.float)

        return fp

'''
#TODO: Check which parameters this fps use and implement it
class TopologicalFingerprint(MolecularFeaturizer):
    """Topological fingerprints.
    """

    def __init__(self,
                 ...):
        """
        Parameters
        ----------
        ...
        """

        self ...

    def _featurize(self, mol: Any) -> np.ndarray:
        """Calculate morgan fingerprint.
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of circular fingerprint.
        """

        try :
            fp = FingerprintMols.FingerprintMol(mol,
                                                ...)
        except Exception as e:
            #print(e)
            print('error in smile: ' + str(mol))
            fp = np.nan
        fp = np.asarray(fp, dtype=np.float)

        return fp
        
'''