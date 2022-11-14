import numpy as np
from rdkit.Chem import Mol, rdMolDescriptors, MACCSkeys, rdmolops
from rdkit.Chem.rdMolDescriptors import GetAtomPairAtomCode

from deepmol.compound_featurization import MolecularFeaturizer


class MorganFingerprint(MolecularFeaturizer):
    """
    Morgan fingerprints.
    Extended Connectivity Circular Fingerprints compute a bag-of-words style
    representation of a molecule by breaking it into local neighborhoods and
    hashing into a bit vector of the specified size.
    """

    def __init__(self, radius: int = 2, size: int = 2048, chiral: bool = False, bonds: bool = True,
                 features: bool = False):
        """
        Initialize a MorganFingerprint object.

        Parameters
        ----------
        radius: int
            The radius of the circular fingerprint.
        size: int
            The size of the fingerprint.
        chiral: bool
            Whether to include chirality in the fingerprint.
        bonds: bool
            Whether to consider bond order in fingerprint generation.
        features: bool
            Whether to use feature information instead of atom information.
        """
        super().__init__()
        self.radius = radius
        self.size = size
        self.chiral = chiral
        self.bonds = bonds
        self.features = features

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Calculate morgan fingerprint for a single molecule.

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
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
            fp = np.empty(self.size, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)
        return fp


class MACCSkeysFingerprint(MolecularFeaturizer):
    """
    MACCS Keys.
    SMARTS-based implementation of the 166 public MACCS keys.
    """

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Calculate MACCSkeys for a single molecule.

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
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
    """
    Calculate layered fingerprint for a single molecule.

    Layer definitions:
        0x01: pure topology
        0x02: bond order
        0x04: atom types
        0x08: presence of rings
        0x10: ring sizes
        0x20: aromaticity
    """

    def __init__(self,
                 layerFlags: int = 4294967295,
                 minPath: int = 1,
                 maxPath: int = 7,
                 fpSize: int = 2048,
                 atomCounts: list = None,
                 branchedPaths: bool = True):
        """
        Initialize a LayeredFingerprint object.

        Parameters
        ----------
        layerFlags: int
            A bit vector specifying which layers to include in the fingerprint.
        minPath: int
            The minimum number of bonds to include in the subgraphs.
        maxPath: int
            The maximum number of bonds to include in the subgraphs.
        fpSize: int
            The size of the fingerprint.
        atomCounts: None
            If provided, this should be a list at least as long as the number of atoms in the molecule.
            It will be used to provide the count of the number of paths that set bits each atom is involved in.
        branchedPaths: bool
            Whether to include branched and unbranched paths in the fingerprint.
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

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Calculate layered fingerprint for a single molecule.

        Parameters
        ----------
        mol: Mol
          RDKit Mol object
        Returns
        -------
        fp: np.ndarray
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
            fp = np.empty(self.fpSize, dtype=float)
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

    def __init__(self,
                 minPath: int = 1,
                 maxPath: int = 7,
                 fpSize: int = 2048,
                 nBitsPerHash: int = 2,
                 useHs: bool = True,
                 tgtDensity: float = 0.0,
                 minSize: int = 128,
                 branchedPaths: bool = True,
                 useBondOrder: bool = True):

        """
        Initialize a RDKFingerprint object.

        Parameters
        ----------
        minPath: int
            The minimum number of bonds to include in the subgraphs.
        maxPath: int
            The maximum number of bonds to include in the subgraphs.
        fpSize: int
            The size of the fingerprint.
        nBitsPerHash: int
            The number of bits to set for each hash.
        useHs: bool
            Whether to include Hs in the subgraphs.
        tgtDensity: float
            Fold the fingerprint until this minimum density has been reached.
        minSize: int
            The minimum size the fingerprint will be folded to when trying to reach tgtDensity.
        branchedPaths: bool
            Whether to include branched and unbranched paths in the fingerprint.
        useBondOrder: bool
            If True, both bond orders will be used in the path hashes
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

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Calculate topological fingerprint for a single molecule.

        Parameters
        ----------
        mol: Mol
          RDKit Mol object
        Returns
        -------
        fp: np.ndarray
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

    def __init__(self,
                 nBits: int = 2048,
                 minLength: int = 1,
                 maxLength: int = 30,
                 nBitsPerEntry: int = 4,
                 includeChirality: bool = False,
                 use2D: bool = True,
                 confId: int = -1):
        """
        Initialize an AtomPairFingerprint object.

        Parameters
        ----------
        nBits: int
            The size of the fingerprint.
        minLength: int
            Minimum distance between atoms to be considered in a pair.
        maxLength: int
            Maximum distance between atoms to be considered in a pair.
        nBitsPerEntry: int
            The number of bits to use in simulating counts.
        includeChirality: bool
            If set, chirality will be used in the atom invariants.
        use2D: bool
            If set, the 2D (topological) distance matrix is used.
        confId: int
            The conformation to use if 3D distances are being used return a pointer to the fingerprint.
        """
        super().__init__()
        self.nBits = nBits
        self.minLength = minLength
        self.maxLength = maxLength
        self.nBitsPerEntry = nBitsPerEntry
        self.includeChirality = includeChirality
        self.use2D = use2D
        self.confId = confId

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Calculate atom pair fingerprint for a single molecule.

        Parameters
        ----------
        mol: Mol
          RDKit Mol object
        Returns
        -------
        fp: np.ndarray
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


class AtomPairFingerprintCallbackHash(MolecularFeaturizer):
    """
    Atom pair fingerprints

    Returns the atom-pair fingerprint for a molecule as an ExplicitBitVect
    """

    def __init__(self,
                 nBits: int = 2048,
                 minLength: int = 1,
                 maxLength: int = 30,
                 includeChirality: bool = False,
                 use2D: bool = True,
                 confId: int = -1):
        """
        Initialize an AtomPairFingerprintCallbackHash object.

        Parameters
        ----------
        nBits: int
            The size of the fingerprint.
        minLength: int
            Minimum distance between atoms to be considered in a pair.
        maxLength: int
            Maximum distance between atoms to be considered in a pair.
        includeChirality: bool
            If set, chirality will be used in the atom invariants.
        use2D: bool
            If set, the 2D (topological) distance matrix is used.
        confId: int
            The conformation to use if 3D distances are being used return a pointer to the fingerprint.
        """
        super().__init__()
        self.nBits = nBits
        self.minLength = minLength
        self.maxLength = maxLength
        self.includeChirality = includeChirality
        self.use2D = use2D
        self.confId = confId

    @staticmethod
    def hash_function(bit, value):
        """
        Hash function for atom pair fingerprint.

        Parameters
        ----------
        bit: int
            The bit to be hashed.
        value: int
            The value to be hashed.
        """
        bit = hash(value) + 0x9e3779b9 + (bit * (2 ** 6)) + (bit / (2 ** 2))
        return bit

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Calculate AtomPairFingerprintCallbackHash for a single molecule.

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
          A numpy array of layered fingerprints.
        """
        try:
            matrix = rdmolops.GetDistanceMatrix(mol)
            fp = [0] * self.nBits
            for at1 in range(mol.GetNumAtoms()):
                for at2 in range(at1 + 1, mol.GetNumAtoms()):
                    atom1 = mol.GetAtomWithIdx(at1)
                    atom2 = mol.GetAtomWithIdx(at2)
                    at1_hash_code = GetAtomPairAtomCode(atom1, includeChirality=self.includeChirality)
                    at2_hash_code = GetAtomPairAtomCode(atom2, includeChirality=self.includeChirality)

                    if self.minLength <= int(matrix[at1][at2]) <= self.maxLength:
                        bit = self.hash_function(0, min(at1_hash_code, at2_hash_code))
                        bit = self.hash_function(bit, matrix[at1][at2])
                        bit = self.hash_function(bit, max(at1_hash_code, at2_hash_code))
                        index = int(bit % self.nBits)
                        fp[index] = 1
        except Exception as e:
            print('error in smile: ' + str(mol))
            fp = np.empty(self.nBits, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp
