import os.path
import random
from typing import Tuple, Union, List

import PIL
import numpy as np
try:
    from IPython.display import display
except ImportError:
    print("IPython not found. Unable to display molecule images.")
from rdkit.Chem import MACCSkeys, Draw
from rdkit.Chem.rdMolDescriptors import GetAtomPairAtomCode

from deepmol.compound_featurization import MolecularFeaturizer

from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
import tempfile
from PIL import Image

from rdkit.Chem import rdMolDescriptors, Mol, rdmolops

from deepmol.compound_featurization._constants import MACCSsmartsPatts
from deepmol.compound_featurization._utils import get_substructure_depiction, svg_text_to_file


class MorganFingerprint(MolecularFeaturizer):
    """
    Morgan fingerprints.
    Extended Connectivity Circular Fingerprints compute a bag-of-words style
    representation of a molecule by breaking it into local neighborhoods and
    hashing into a bit vector of the specified size.
    """

    def __init__(self, radius: int = 2, size: int = 2048, chiral: bool = False, bonds: bool = True,
                 features: bool = False, **kwargs):
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
        super().__init__(**kwargs)
        self.radius = radius
        self.size = size
        self.chiral = chiral
        self.bonds = bonds
        self.features = features
        self.feature_names = [f'morgan_{i}' for i in range(self.size)]

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
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
                                                            self.radius,
                                                            nBits=self.size,
                                                            useChirality=self.chiral,
                                                            useBondTypes=self.bonds,
                                                            useFeatures=self.features)
        fp = np.asarray(fp, dtype=np.float32)
        return fp

    def draw_bit(self, mol: Mol, bit: int, molSize: Tuple[int, int] = (450, 200), file_path: str = None) -> str:
        """
        Draw a molecule with a Morgan fingerprint bit highlighted.

        Parameters
        ----------
        mol: Mol
            Molecule to draw.
        bit: int
            Bit to highlight.
        molSize: Tuple[int, int]
            Size of the molecule.
        file_path: str
            Path to save the image.

        Returns
        -------
        str
            The molecule in SVG format.
        """
        if mol is None:
            raise ValueError('Molecule is None! Please insert a valid molecule')

        info = {}
        rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, self.radius,
                                                       nBits=self.size,
                                                       useChirality=self.chiral,
                                                       useBondTypes=self.bonds,
                                                       useFeatures=self.features,
                                                       bitInfo=info)

        if bit not in info.keys():
            self.logger.info(f'Bits ON: {list(info.keys())}')
            raise ValueError('Bit is off! Bits ON: %s' % (list(info.keys())))

        self.logger.info('Bit %d with %d hits!' % (bit, len(info[bit])))

        aid, rad = info[bit][0]

        depiction = get_substructure_depiction(mol, aid, rad, molSize=molSize)

        if file_path is not None:
            svg_text_to_file(depiction.data, file_path)

        return depiction

    def draw_bits(self, mol: Mol, bit_indexes: Union[int, str, List[int]],
                  file_path: str = None) -> str:
        """
        Draw a molecule with a Morgan fingerprint bit highlighted.

        Parameters
        ----------
        mol: Mol
            Molecule to draw.
        bit_indexes: Union[int, str, List[int]]
            Bit to highlight. If int, only one bit is highlighted. If list, all the bits in the list are highlighted.
            If 'ON', all the bits ON are highlighted.
        file_path : str
            Path to save the image.

        Returns
        -------
        str
        """

        if mol is None:
            raise ValueError('Molecule is None! Please insert a valid molecule')

        bi = {}

        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
                                                            radius=self.radius,
                                                            nBits=self.size,
                                                            useChirality=self.chiral,
                                                            useBondTypes=self.bonds,
                                                            useFeatures=self.features,
                                                            bitInfo=bi)

        if isinstance(bit_indexes, int):
            if bit_indexes not in bi.keys():
                self.logger.info(f'Bits ON: {bi.keys()}')
                raise ValueError('Bit is off! Select a on bit')
            svg_text = Draw.DrawMorganBit(mol, bit_indexes, bi)
            if file_path is not None:
                svg_text_to_file(svg_text, file_path)
            return svg_text

        elif isinstance(bit_indexes, list):
            bits_on = []
            for b in bit_indexes:
                if b in bi.keys():
                    bits_on.append(b)
                else:
                    self.logger.info('Bit %d is off!' % (b))
            if len(bits_on) == 0:
                raise ValueError('All the selected bits are off! Select on bits!')
            elif len(bits_on) != len(bit_indexes):
                self.logger.info(f'Using only bits ON: {bits_on}')
            tpls = [(mol, x, bi) for x in bits_on]
            svg_text = Draw.DrawMorganBits(tpls, molsPerRow=5, legends=['bit_' + str(x) for x in bits_on])

            if file_path is not None:
                svg_text_to_file(svg_text, file_path)

            return svg_text
        elif bit_indexes == 'ON':
            tpls = [(mol, x, bi) for x in fp.GetOnBits()]
            svg_text = Draw.DrawMorganBits(tpls, molsPerRow=5, legends=[str(x) for x in fp.GetOnBits()])
            if file_path is not None:
                svg_text_to_file(svg_text, file_path)
            return svg_text

        else:
            raise ValueError('Bits must be integer, list of integers or ON!')


class MACCSkeysFingerprint(MolecularFeaturizer):
    """
    MACCS Keys.
    SMARTS-based implementation of the 166 public MACCS keys.
    """

    def __init__(self, **kwargs):
        """
        Initialize a MACCSkeysFingerprint object.
        """
        super().__init__(**kwargs)
        self.feature_names = [f'maccs_{i}' for i in range(167)]

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
        fp = MACCSkeys.GenMACCSKeys(mol)
        fp = np.asarray(fp, dtype=np.float32)
        return fp

    def draw_bit(self, mol: Mol, bit_index: int, file_path: str = None) -> PIL.Image.Image:
        """
        Draw a molecule with a MACCS key highlighted.

        Parameters
        ----------
        mol: Mol
            Molecule to draw.
        bit_index: int
            Index of the MACCS key to highlight.
        file_path: str
            Path to save the image to. If None, the image is not saved.

        Returns
        -------
        im: PIL.Image.Image
            Image of the molecule with the MACCS key highlighted.
        """

        if mol is None:
            raise ValueError('Molecule cannot be None!')

        if bit_index not in MACCSsmartsPatts.keys():
            raise ValueError('Bit index must be between 1 and 166!')

        smart = MACCSsmartsPatts[bit_index][0]
        if "?" in smart:
            raise ValueError('Bit %d cannot be drawn!' % (bit_index))

        patt = Chem.MolFromSmarts(smart)

        if mol.HasSubstructMatch(patt):
            hit_ats = mol.GetSubstructMatches(patt)
            bond_lists = []
            for i, hit_at in enumerate(hit_ats):
                hit_at = list(hit_at)
                bond_list = []
                for bond in patt.GetBonds():
                    a1 = hit_at[bond.GetBeginAtomIdx()]
                    a2 = hit_at[bond.GetEndAtomIdx()]
                    bond_list.append(mol.GetBondBetweenAtoms(a1, a2).GetIdx())
                bond_lists.append(bond_list)

            colours = []
            for i in range(len(hit_ats)):
                colours.append((random.random(), random.random(), random.random()))
            atom_cols = {}
            bond_cols = {}
            atom_list = []
            bond_list = []
            for i, (hit_atom, hit_bond) in enumerate(zip(hit_ats, bond_lists)):
                hit_atom = list(hit_atom)
                for at in hit_atom:
                    atom_cols[at] = colours[i]
                    atom_list.append(at)
                for bd in hit_bond:
                    bond_cols[bd] = colours[i]
                    bond_list.append(bd)
            d = rdMolDraw2D.MolDraw2DCairo(500, 500)
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=atom_list,
                                               highlightAtomColors=atom_cols,
                                               highlightBonds=bond_list,
                                               highlightBondColors=bond_cols)

            d.FinishDrawing()
            if file_path is None:
                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    d.WriteDrawingText(tmp_dir_name + 'mol.png')
                    im = Image.open(tmp_dir_name + 'mol.png')
                    return im
            elif ".png" in file_path.lower():
                d.WriteDrawingText(file_path)
                im = Image.open(file_path)
                return im
            else:
                raise ValueError('File path must end with .png!')
        else:
            fp = MACCSkeys.GenMACCSKeys(mol)
            fp = np.asarray(fp, dtype=np.float32)
            bits_on = np.where(fp == 1)[0]
            self.logger.info(f'Pattern does not match molecule! Active bits: {bits_on}')


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
                 branchedPaths: bool = True,
                 **kwargs):
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
        super().__init__(**kwargs)
        if atomCounts is None:
            atomCounts = []
        self.layerFlags = layerFlags
        self.minPath = minPath
        self.maxPath = maxPath
        self.fpSize = fpSize
        self.atomCounts = atomCounts
        self.branchedPaths = branchedPaths
        self.feature_names = [f'layered_{i}' for i in range(self.fpSize)]

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
        fp = rdmolops.LayeredFingerprint(mol,
                                         layerFlags=self.layerFlags,
                                         minPath=self.minPath,
                                         maxPath=self.maxPath,
                                         fpSize=self.fpSize,
                                         atomCounts=self.atomCounts,
                                         branchedPaths=self.branchedPaths)
        fp = np.asarray(fp, dtype=np.float32)
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
                 useBondOrder: bool = True,
                 **kwargs):
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
        super().__init__(**kwargs)
        self.minPath = minPath
        self.maxPath = maxPath
        self.fpSize = fpSize
        self.nBitsPerHash = nBitsPerHash
        self.useHs = useHs
        self.tgtDensity = tgtDensity
        self.minSize = minSize
        self.branchedPaths = branchedPaths
        self.useBondOrder = useBondOrder
        self.feature_names = [f'rdk_{i}' for i in range(self.fpSize)]

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
        fp = np.asarray(fp, dtype=np.float32)
        return fp

    def draw_bits(self, mol: Mol, bits: Union[int, str, List[int]], file_path: str = None) -> str:
        """
        Draw a molecule with a RDK fingerprint bit highlighted.

        Parameters
        ----------
        mol: Mol
            Molecule to draw.
        bits: Union[int, str, List[int]]
            Bit to highlight. If int, the bit to highlight. If str, the name of the bit to highlight.
            If 'ON', all the bits ON are highlighted.
        file_path: str
            Path to save the image. If None, the image is not saved.
        Returns
        -------
        Str
            The molecule with the fingerprint bits.
        """
        rdkbit = {}
        fp = rdmolops.RDKFingerprint(mol,
                                     minPath=self.minPath,
                                     maxPath=self.maxPath,
                                     fpSize=self.fpSize,
                                     nBitsPerHash=self.nBitsPerHash,
                                     useHs=self.useHs,
                                     tgtDensity=self.tgtDensity,
                                     minSize=self.minSize,
                                     branchedPaths=self.branchedPaths,
                                     useBondOrder=self.useBondOrder,
                                     bitInfo=rdkbit)
        if isinstance(bits, int):
            if bits not in rdkbit.keys():
                self.logger.info(f'Bits ON: {list(rdkbit.keys())}')
                raise ValueError(f'Bit is off! Select a on bit. Bits ON: {list(rdkbit.keys())}')
            svg_text = Draw.DrawRDKitBit(mol, bits, rdkbit)
            if file_path is not None:
                svg_text_to_file(svg_text, file_path)
            return svg_text

        elif isinstance(bits, list):
            bits_on = []
            for bit in bits:
                if bit in rdkbit.keys():
                    bits_on.append(bit)
                else:
                    self.logger.info(f'Bit {bit} is off! Select a on bit. Bits ON: {list(rdkbit.keys())} ')
            if len(bits_on) == 0:
                raise ValueError('All the selected bits are off! Select on bits! Bit is off! Select a on bit. Bits '
                                 f'ON: {bits_on}')
            elif len(bits_on) != len(bits):
                self.logger.info(f'Bits ON: {bits_on}')

            tpls = [(mol, x, rdkbit) for x in bits_on]
            svg_text = Draw.DrawRDKitBits(tpls, molsPerRow=5, legends=['bit_' + str(x) for x in bits_on])
            if file_path is not None:
                svg_text_to_file(svg_text, file_path)
            return svg_text

        elif bits == 'ON':
            tpls = [(mol, x, rdkbit) for x in fp.GetOnBits()]
            svg_text = Draw.DrawRDKitBits(tpls, molsPerRow=5, legends=[str(x) for x in fp.GetOnBits()])
            if file_path is not None:
                svg_text_to_file(svg_text, file_path)
            return svg_text

        else:
            raise ValueError('Bits must be integer, list of integers or ON!')

    def draw_bit(self, mol: Mol,
                 bit: int,
                 folder_path: str = None,
                 molSize: Tuple[int, int] = (450, 200)):
        """
        Draw a molecule with a RDK fingerprint bit highlighted.

        Parameters
        ----------
        mol: Mol
            Molecule to draw.
        bit: int
            Bit to highlight.
        folder_path: str
            Path for the folder to save images.
        molSize: Tuple[int, int]
            Size of the molecule.

        Returns
        -------
        Images
            The molecule with the fingerprint bit highlighted.
        """
        if mol is None:
            raise ValueError('Mol is None!')

        info = {}
        rdmolops.RDKFingerprint(mol,
                                minPath=self.minPath,
                                maxPath=self.maxPath,
                                fpSize=self.fpSize,
                                nBitsPerHash=self.nBitsPerHash,
                                useHs=self.useHs,
                                tgtDensity=self.tgtDensity,
                                minSize=self.minSize,
                                branchedPaths=self.branchedPaths,
                                useBondOrder=self.useBondOrder,
                                bitInfo=info)

        if bit not in info.keys():
            self.logger.info(f'Bits ON: {info.keys()}')
            raise ValueError(f'Bit is off! Select a on bit. Bits ON: {list(info.keys())}')

        self.logger.info('Bit %d with %d hits!' % (bit, len(info[bit])))

        images = []
        for i in range(len(info[bit])):
            d = rdMolDraw2D.MolDraw2DCairo(molSize[0], molSize[1])
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightBonds=info[bit][i])
            d.FinishDrawing()
            if folder_path is None:
                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    file_to_save_image = os.path.join(tmp_dir_name, f'mol_{i}.png')
                    d.WriteDrawingText(file_to_save_image)
                    im = Image.open(file_to_save_image)
                    images.append(im)
                    im.close()
            else:
                os.makedirs(folder_path, exist_ok=True)
                file_to_save_image = os.path.join(folder_path, f'mol_{i}.png')
                d.WriteDrawingText(file_to_save_image)
                im = Image.open(file_to_save_image)
                images.append(im)
                im.close()
        return display(*images)


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
                 confId: int = -1,
                 **kwargs):
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
        super().__init__(**kwargs)
        self.nBits = nBits
        self.minLength = minLength
        self.maxLength = maxLength
        self.nBitsPerEntry = nBitsPerEntry
        self.includeChirality = includeChirality
        self.use2D = use2D
        self.confId = confId
        self.feature_names = [f'atom_pair_{i}' for i in range(self.nBits)]

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
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol,
                                                                    nBits=self.nBits,
                                                                    minLength=self.minLength,
                                                                    maxLength=self.maxLength,
                                                                    nBitsPerEntry=self.nBitsPerEntry,
                                                                    includeChirality=self.includeChirality,
                                                                    use2D=self.use2D,
                                                                    confId=self.confId)
        fp = np.asarray(fp, dtype=np.float32)
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
                 confId: int = -1,
                 **kwargs):
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
        super().__init__(**kwargs)
        self.nBits = nBits
        self.minLength = minLength
        self.maxLength = maxLength
        self.includeChirality = includeChirality
        self.use2D = use2D
        self.confId = confId
        self.feature_names = [f'atom_pair_hash_{i}' for i in range(self.nBits)]

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
        fp = np.asarray(fp, dtype=np.float32)
        return fp
