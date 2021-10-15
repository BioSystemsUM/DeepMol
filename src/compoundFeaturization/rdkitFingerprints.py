from src.compoundFeaturization.baseFeaturizer import MolecularFeaturizer
from rdkit.Chem import rdMolDescriptors, MACCSkeys, rdmolops
import numpy as np
from typing import Any


# TODO: final output of featurization (<class 'numpy.ndarray'> of  <class 'numpy.ndarray'>) and
#  feature selection (<class 'numpy.ndarray'> of <class 'numpy.ndarray'>) but when printed are not the same ?????
#  even the shape of the output is different
class MorganFingerprint(MolecularFeaturizer):
    """Morgan fingerprints.
    Extended Connectivity Circular Fingerprints compute a bag-of-words style
    representation of a molecule by breaking it into local neighborhoods and
    hashing into a bit vector of the specified size.
    """

    def __init__(self, radius: int = 2, size: int = 2048, chiral: bool = False, bonds: bool = True,
                 features: bool = False):
        """
        Parameters
        ----------
        radius: int, optional (default 2)
          Fingerprint radius.
        size: int, optional (default 2048)
          Length of generated bit vector.
        chiral: bool, optional (default False)
          Whether to consider chirality in fingerprint generation.
        bonds: bool, optional (default True)
          Whether to consider bond order in fingerprint generation.
        features: bool, optional (default False)
          Whether to use feature information instead of atom information;
        """

        super().__init__()
        self.radius = radius
        self.size = size
        self.chiral = chiral
        self.bonds = bonds
        self.features = features

    def _featurize(self, mol: Any) -> np.ndarray:
        """Calculate morgan fingerprint for a single molecule.fre
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of circular fingerprint.
        """

        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
                                                                self.radius,
                                                                nBits=self.size,
                                                                useChirality=self.chiral,
                                                                useBondTypes=self.bonds,
                                                                useFeatures=self.features)
        except Exception as e:
            print('error in smile: ' + str(mol))
            # fp = np.nan
            fp = np.empty(self.size, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp


class MACCSkeysFingerprint(MolecularFeaturizer):
    """MACCS Keys.
    SMARTS-based implementation of the 166 public MACCS keys.
    """

    def _featurize(self, mol: Any) -> np.ndarray:
        """Calculate MACCSkeys for a single molecule.
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of MACCSkeys.
        """

        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
        except Exception as e:
            print('error in smile: ' + str(mol))
            fp = np.empty(167, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp


class LayeredFingerprint(MolecularFeaturizer):
    # TODO: Comments
    """
    ...

    Layer definitions:
        0x01: pure topology
        0x02: bond order
        0x04: atom types
        0x08: presence of rings
        0x10: ring sizes
        0x20: aromaticity
    """

    def __init__(self, layerFlags: int = 4294967295, minPath: int = 1, maxPath: int = 7, fpSize: int = 2048,
                 atomCounts=None, branchedPaths: bool = True):
        """
        Parameters
        ----------
        layerFlags: (optional)
            which layers to include in the fingerprint See below for definitions. Defaults to all.
        minPath: (optional)
            minimum number of bonds to include in the subgraphs Defaults to 1.
        maxPath: (optional)
            maximum number of bonds to include in the subgraphs Defaults to 7.
        fpSize: (optional)
            number of bits in the fingerprint Defaults to 2048.
        atomCounts: (optional)
            if provided, this should be a list at least as long as the number of atoms in the molecule.
            It will be used to provide the count of the number of paths that set bits each atom is involved in.
        branchedPaths: (optional)
            if set both branched and unbranched paths will be used in the fingerprint. Defaults to True.
        """

        super().__init__()
        if atomCounts is None:
            atomCounts = []
        self.layerFlags = layerFlags
        self.minPath = minPath
        self.maxPath = maxPath
        self.fpSize = fpSize
        self.atomCounts = atomCounts
        self.branchedPaths = branchedPaths

    def _featurize(self, mol: Any) -> np.ndarray:
        """Calculate layered fingerprint for a single molecule.
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of layered fingerprints.
        """

        try:
            fp = rdmolops.LayeredFingerprint(mol,
                                             layerFlags=self.layerFlags,
                                             minPath=self.minPath,
                                             maxPath=self.maxPath,
                                             fpSize=self.fpSize,
                                             atomCounts=self.atomCounts,
                                             branchedPaths=self.branchedPaths)
        except Exception as e:
            print('error in smile: ' + str(mol))
            fp = np.empty(self.fpsize, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp


class RDKFingerprint(MolecularFeaturizer):
    """
    RDKit topological fingerprints

    This algorithm functions by find all subgraphs between minPath and maxPath in length. For each subgraph:

        A hash is calculated.

        The hash is used to seed a random-number generator

        _nBitsPerHash_ random numbers are generated and used to set the corresponding bits in the fingerprint

    """

    def __init__(self, minPath: int = 1, maxPath: int = 7, fpSize: int = 2048, nBitsPerHash: int = 2,
                 useHs: bool = True, tgtDensity: float = 0.0, minSize: int = 128, branchedPaths: bool = True,
                 useBondOrder: bool = True):

        """
        Parameters
        ----------
        minPath: (optional)
            minimum number of bonds to include in the subgraphs Defaults to 1.
        maxPath: (optional)
            maximum number of bonds to include in the subgraphs Defaults to 7.
        fpSize: (optional)
            number of bits in the fingerprint Defaults to 2048.
        nBitsPerHash: (optional)
            number of bits to set per path Defaults to 2.
        useHs: (optional)
            include paths involving Hs in the fingerprint if the molecule has explicit Hs. Defaults to True.
        tgtDensity: (optional)
            fold the fingerprint until this minimum density has been reached Defaults to 0.
        minSize: (optional)
            the minimum size the fingerprint will be folded to when trying to reach tgtDensity Defaults to 128.
        branchedPaths: (optional)
            if set both branched and unbranched paths will be used in the fingerprint. Defaults to True.
        useBondOrder: (optional)
            if set both bond orders will be used in the path hashes Defaults to True.
        """

        super().__init__()
        self.minPath = minPath
        self.maxPath = maxPath
        self.fpSize = fpSize
        self.nBitsPerHash = nBitsPerHash
        self.useHs = useHs
        self.tgtDensity = tgtDensity
        self.minSize = minSize
        self.branchedPaths = branchedPaths
        self.useBondOrder = useBondOrder

    def _featurize(self, mol: Any) -> np.ndarray:
        """Calculate topological fingerprint for a single molecule.
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of layered fingerprints.
        """

        try:
            fp = rdmolops.RDKFingerprint(mol,
                                         minPath=self.minPath,
                                         maxPath=self.maxPath,
                                         fpSize=self.fpSize,
                                         nBitsPerHash=self.nBitsPerHash,
                                         useHs=self.useHs,
                                         tgtDensity=self.tgtDensity,
                                         minSize=self.minSize,
                                         branchedPaths=self.branchedPaths,
                                         useBondOrder=self.useBondOrder)

        except Exception as e:
            print('error in smile: ' + str(mol))
            fp = np.empty(self.fpSize, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp


class AtomPairFingerprint(MolecularFeaturizer):
    """
    Atom pair fingerprints

    Returns the atom-pair fingerprint for a molecule as an ExplicitBitVect

    """

    def __init__(self, nBits: int = 2048, minLength: int = 1, maxLength: int = 30, nBitsPerEntry: int = 4,
                 includeChirality: bool = False, use2D: bool = True, confId: int = -1):
        """
        Parameters
        ----------
        nBits: (optional)
            ...
        """

        super().__init__()
        self.nBits = nBits
        self.minLength = minLength
        self.maxLength = maxLength
        self.nBitsPerEntry = nBitsPerEntry
        self.includeChirality = includeChirality
        self.use2D = use2D
        self.confId = confId

    def _featurize(self, mol: Any) -> np.ndarray:
        """Calculate atom pair fingerprint for a single molecule.
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of layered fingerprints.
        """

        try:
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol,
                                                                        nBits=self.nBits,
                                                                        minLength=self.minLength,
                                                                        maxLength=self.maxLength,
                                                                        nBitsPerEntry=self.nBitsPerEntry,
                                                                        includeChirality=self.includeChirality,
                                                                        use2D=self.use2D,
                                                                        confId=self.confId)

        except Exception as e:
            print('error in smile: ' + str(mol))
            fp = np.empty(self.nBits, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp

# TODO: add rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect; Generate.Gen2DFingerprint;
# rdReducedGraphs.GetErGFingerprint;
