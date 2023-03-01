from typing import List, Dict, Any

import numpy as np
from deepchem.feat import ConvMolFeaturizer, WeaveFeaturizer, MolGraphConvFeaturizer, CoulombMatrix, CoulombMatrixEig, \
    SmilesToImage, SmilesToSeq, MolGanFeaturizer, GraphMatrix
from deepchem.feat.graph_data import GraphData
from deepchem.feat.mol_graphs import ConvMol, WeaveMol
from deepchem.utils import ConformerGenerator
from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles

from deepmol.compound_featurization import MolecularFeaturizer
from deepmol.compound_featurization._utils import get_conformers, get_dictionary_from_smiles
from deepmol.datasets import Dataset
from deepmol.loggers.logger import Logger


class ConvMolFeat(MolecularFeaturizer):
    """
    Duvenaud graph convolution, adapted from deepchem
    (https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#convmolfeaturizer).
    Vector of descriptors for each atom in a molecule.
    The featurizers computes that vector of local descriptors.

    References:
    Duvenaud, David K., et al. "Convolutional networks on graphs for learning molecular fingerprints."
    Advances in neural information processing systems. 2015.
    """

    def __init__(self,
                 master_atom: bool = False,
                 use_chirality: bool = False,
                 atom_properties: List[str] = None,
                 per_atom_fragmentation: bool = False,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        master_atom: bool
            If True, create a fake atom with bonds to every other atom.
        use_chirality: bool
            If True, include chirality information.
        atom_properties: List[str]
            List of atom properties to use as additional atom-level features in the larger molecular feature.
        per_atom_fragmentation: bool
            If True, then multiple "atom-depleted" versions of each molecule will be created.
        kwargs:
            Additional arguments for the base class.
        """
        super().__init__(**kwargs)
        if atom_properties is None:
            atom_properties = []
        self.master_atom = master_atom
        self.use_chirality = use_chirality
        self.atom_properties = atom_properties
        self.per_atom_fragmentation = per_atom_fragmentation
        self.feature_names = ['conv_mol_feat']

    def _featurize(self, mol: Mol) -> ConvMol:
        """
        Featurizes a single molecule.

        Parameters
        ----------
        mol: Mol
            Molecule to featurize.

        Returns
        -------
        feature: ConvMol
            The ConvMol features of the molecule.
        """
        # featurization process using DeepChem ConvMolFeaturizer
        feature = ConvMolFeaturizer(
            master_atom=self.master_atom,
            use_chirality=self.use_chirality,
            atom_properties=self.atom_properties,
            per_atom_fragmentation=self.per_atom_fragmentation).featurize([mol])

        assert feature[0].atom_features is not None
        return feature[0]


class WeaveFeat(MolecularFeaturizer):
    """
    Weave convolution featurization, adapted from deepchem
    (https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#weavefeaturizer).
    Require a quadratic matrix of interaction descriptors for each pair of atoms.

    References:
    Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
    Journal of computer-aided molecular design 30.8 (2016): 595-608.
    """

    def __init__(self,
                 graph_distance: bool = True,
                 explicit_h: bool = False,
                 use_chirality: bool = False,
                 max_pair_distance: int = None,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        graph_distance: bool
            If True, use graph distance for distance features. Otherwise, use Euclidean distance. Molecules invoked must
            have valid conformer information if this option is set.
        explicit_h: bool
            If true, model hydrogens in the molecule.
        use_chirality: bool
            If True, use chiral information in the featurization.
        max_pair_distance: int
            Maximum graph distance at which pair features are computed.
        kwargs:
            Additional arguments for the base class.
        """
        super().__init__(**kwargs)
        self.graph_distance = graph_distance
        self.explicit_h = explicit_h
        self.use_chirality = use_chirality
        self.max_pair_distance = max_pair_distance
        self.feature_names = ['weave_feat']

    def _featurize(self, mol: Mol) -> WeaveMol:
        """
        Featurizes a single molecule.

        Parameters
        ----------
        mol: Mol
            Molecule to featurize.

        Returns
        -------
        feature: WeaveMol
            The WeaveMol features of the molecule.
        """
        # featurization process using DeepChem WeaveFeaturizer
        feature = WeaveFeaturizer(
            graph_distance=self.graph_distance,
            explicit_H=self.explicit_h,
            use_chirality=self.use_chirality,
            max_pair_distance=self.max_pair_distance).featurize([mol])

        assert feature[0].get_atom_features() is not None

        return feature[0]


class MolGanFeat(MolecularFeaturizer):
    """
    Featurizer for MolGAN de-novo molecular generation model, adapted from deepchem
    (https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html?highlight=CGCNN#molganfeaturizer).
    It is wrapper for two matrices containing atom and bond type information.

    References:
    Nicola De Cao et al. “MolGAN: An implicit generative model for small molecular graphs” (2018),
    https://arxiv.org/abs/1805.11973
    """

    def __init__(self,
                 max_atom_count: int = 9,
                 kekulize: bool = True,
                 bond_labels: List[Any] = None,
                 atom_labels: List[int] = None,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        max_atom_count: int
            Maximum number of atoms used for the adjacency matrix creation.
        kekulize: bool
            If True, kekulize the molecule.
        bond_labels: List[Any]
            List of bond types used for the adjacency matrix creation.
        atom_labels: List[int]
            List of atomic numbers used for the adjacency matrix creation.
        kwargs:
            Additional arguments for the base class.
        """
        super().__init__(**kwargs)
        self.max_atom_count = max_atom_count
        self.kekulize = kekulize
        self.bond_labels = bond_labels
        self.atom_labels = atom_labels
        self.feature_names = ['mol_gan_feat']

    def _featurize(self, mol: Mol) -> GraphMatrix:
        """
        Featurizes a single molecule.

        Parameters
        ----------
        mol: Mol
            Molecule to featurize.

        Returns
        -------
        feature: WeaveMol
            The WeaveMol features of the molecule.
        """
        # featurization process using DeepChem MolGanFeat
        feature = MolGanFeaturizer(max_atom_count=self.max_atom_count,
                                   kekulize=self.kekulize,
                                   bond_labels=self.bond_labels,
                                   atom_labels=self.atom_labels).featurize(mol)

        assert feature[0].adjacency_matrix is not None

        return feature[0]


class MolGraphConvFeat(MolecularFeaturizer):
    """
    Featurizer of general graph convolution networks for molecules.
    Adapted from deepchem:
    (https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#molgraphconvfeaturizer)

    References:
    Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
    Journal of computer-aided molecular design 30.8 (2016):595-608.
    """

    def __init__(self,
                 use_edges: bool = False,
                 use_chirality: bool = False,
                 use_partial_charge: bool = False,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        use_edges: bool
            If True, use edge features.
        use_chirality: bool
            If True, use chirality information.
        use_partial_charge: bool
            If True, use partial charge information.
        kwargs:
            Additional arguments for the base class.
        """
        super().__init__(**kwargs)
        self.use_edges = use_edges
        self.use_chirality = use_chirality
        self.use_partial_charge = use_partial_charge
        self.feature_names = ['mol_graph_conv_feat']

    def _featurize(self, mol: Mol) -> GraphData:
        """
        Featurizes a single molecule.

        Parameters
        ----------
        mol: Mol
            Molecule to featurize.

        Returns
        -------
        feature: GraphData
            The GraphData features of the molecule.
        """
        # featurization process using DeepChem MolGraphConvFeaturizer
        feature = MolGraphConvFeaturizer(
            use_edges=self.use_edges,
            use_chirality=self.use_chirality,
            use_partial_charge=self.use_partial_charge).featurize([mol])

        if feature[0].node_features is None:
            raise Exception

        return feature[0]


class CoulombFeat(MolecularFeaturizer):
    """
    Calculate coulomb matrices for molecules.
    Adapted from deepchem (https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#coulombmatrix).

    References:
    Montavon, Grégoire, et al. "Learning invariant representations of molecules for atomization energy prediction."
    Advances in neural information processing systems. 2012.
    """

    def __init__(self,
                 max_atoms: int,
                 remove_hydrogens: bool = False,
                 randomize: bool = False,
                 upper_tri: bool = False,
                 n_samples: int = 1,
                 max_conformers: int = 1,
                 seed: int = None,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        max_atoms: int
            The maximum number of atoms expected for molecules this featurizers will process.
        remove_hydrogens: bool
            If True, remove hydrogens before processing them.
        randomize: bool
            If True, randomize Coulomb matrices. Default to False.
        upper_tri: bool
            Generate only upper triangle part of Coulomb matrices.
        n_samples: int
            If 'randomize' is set to True, the number of random samples to draw.
        max_conformers: int
            Maximum number of conformers.
        seed: int
            Random seed to use.
        kwargs:
            Additional arguments for the base class.
        """
        super().__init__(**kwargs)
        self.max_atoms = max_atoms
        self.remove_hydrogens = remove_hydrogens
        self.randomize = randomize
        self.upper_tri = upper_tri
        self.n_samples = n_samples
        self.max_conformers = max_conformers
        if seed is not None:
            seed = int(seed)
        self.seed = seed
        self.feature_names = ['coulomb_feat']

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Featurizes a single molecule.

        Parameters
        ----------
        mol: Mol
            Molecule to featurize.

        Returns
        -------
        feature: np.ndarray
            Array of features.
        """
        generator = ConformerGenerator(max_conformers=self.max_conformers)
        new_conformers = get_conformers([mol], generator)

        # featurization process using DeepChem CoulombMatrix
        featurizer = CoulombMatrix(
            max_atoms=self.max_atoms,
            remove_hydrogens=self.remove_hydrogens,
            randomize=self.randomize,
            upper_tri=self.upper_tri,
            n_samples=self.n_samples,
            seed=self.seed)

        feature = featurizer(new_conformers)

        if feature[0].size == 0:
            raise Exception

        return feature[0]


class CoulombEigFeat(MolecularFeaturizer):
    """
    Calculate the eigen values of Coulomb matrices for molecules.
    Adapted from deepchem (https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#coulombmatrixeig).

    References:
    Montavon, Grégoire, et al. "Learning invariant representations of molecules for atomization energy prediction."
    Advances in neural information processing systems. 2012.
    """

    def __init__(self,
                 max_atoms: int,
                 remove_hydrogens: bool = False,
                 randomize: bool = False,
                 n_samples: int = 1,
                 max_conformers: int = 1,
                 seed: int = None,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        max_atoms: int
            The maximum number of atoms expected for molecules this featurizers will process.
        remove_hydrogens: bool
            If True, remove hydrogens before processing them.
        randomize: bool
            If True, randomize Coulomb matrices.
        n_samples: int
            If 'randomize' is set to True, the number of random samples to draw.
        max_conformers: int
            maximum number of conformers.
        seed: int
            Random seed to use.
        kwargs:
            Additional arguments for the base class.
        """
        super().__init__(**kwargs)
        self.max_atoms = max_atoms
        self.remove_hydrogens = remove_hydrogens
        self.randomize = randomize
        self.n_samples = n_samples
        if seed is not None:
            seed = int(seed)
        self.seed = seed
        self.max_conformers = max_conformers
        self.feature_names = ['coulomb_eig_feat']

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Featurizes a single molecule.

        Parameters
        ----------
        mol: Mol
            Molecule to featurize.
        Returns
        -------
        feature: np.ndarray
            Array of features.
        """
        generator = ConformerGenerator(max_conformers=self.max_conformers)

        # TO USE in case to add option for the software to find the parameter max_atoms
        # maximum_number_atoms = find_maximum_number_atoms(new_smiles)

        new_conformers = get_conformers([mol], generator)
        # featurization process using DeepChem CoulombMatrixEig
        featurizer = CoulombMatrixEig(
            max_atoms=self.max_atoms,
            remove_hydrogens=self.remove_hydrogens,
            randomize=self.randomize,
            n_samples=self.n_samples,
            seed=self.seed)

        feature = featurizer(new_conformers)

        if feature[0].size == 0:
            raise Exception

        return feature[0]


class SmileImageFeat(MolecularFeaturizer):
    """
    Converts SMILE string to image.
    Adapted from deepchem (https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#smilestoimage).

    References:
    Goh, Garrett B., et al. "Using rule-based labels for weak supervised learning: a ChemNet for transferable chemical
    property prediction."
    Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.
    """

    def __init__(self,
                 img_size: int = 80,
                 res: float = 0.5,
                 max_len: int = 250,
                 img_spec: str = "std",
                 **kwargs) -> None:
        """
        Parameters
        ----------
        img_size: int
            Size of the image tensor.
        res: float
            Displays the resolution of each pixel in Angstrom.
        max_len: int
            Maximum allowed length of SMILES string.
        img_spec: str
            Indicates the channel organization of the image tensor.
        kwargs:
            Additional arguments for the base class.
        """
        super().__init__(**kwargs)
        if img_spec not in ["std", "engd"]:
            raise ValueError(
                "Image mode must be one of the std or engd. {} is not supported".format(img_spec))
        self.img_size = img_size
        self.max_len = max_len
        self.res = res
        self.img_spec = img_spec
        self.embed = int(img_size * res / 2)
        self.feature_names = ['smile_image_feat']

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Featurizes a single molecule.

        Parameters
        ----------
        mol: Mol
            Molecule to featurize.

        Returns
        -------
        features: np.ndarray
            Array of features.
        """
        # featurization process using DeepChem SmilesToImage
        feats = SmilesToImage(
            img_size=self.img_size,
            max_len=self.max_len,
            res=self.res,
            img_spec=self.img_spec).featurize([mol])

        # identify which rows did not get featurized
        if len(feats[0]) == 0:
            raise Exception

        return feats


class SmilesSeqFeat:
    """
    Takes SMILES strings and turns into a sequence.
    Adapted from deepchem (https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#smilestoseq).

    References:
    Goh, Garrett B., et al. "Using rule-based labels for weak supervised learning: a ChemNet for transferable chemical
    property prediction."
    Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.
    """

    def __init__(self,
                 char_to_idx: Dict[str, int] = None,
                 max_len: int = 250,
                 pad_len: int = 10) -> None:
        """
        Parameters
        ----------
        char_to_idx: Dict
            Dictionary containing character to index mappings for unique characters.
        max_len: int
            Maximum allowed length of the SMILES string.
        pad_len: int
            Amount of padding to add on either side of the SMILES seq.
        """
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        self.pad_len = pad_len
        self.feature_names = ['smiles_seq_feat']
        self.logger = Logger()

    def featurize(self, dataset: Dataset) -> Dataset:
        """
        Featurizes a single molecule.

        Parameters
        ----------
        dataset: Dataset
            Dataset to featurize.

        Returns
        -------
        dataset: Dataset
            Featurized dataset.
        """
        # Getting the dictionary if it is None
        if self.char_to_idx is None:
            if isinstance(dataset.mols[0], Mol):
                smiles = [MolToSmiles(mol) for mol in dataset.mols]
            elif isinstance(dataset.mols[0], str):
                smiles = dataset.mols
            else:
                smiles = None

            self.char_to_idx = get_dictionary_from_smiles(smiles, self.max_len)

        dataset.dictionary = self.char_to_idx

        # obtain new SMILE's strings
        if isinstance(dataset.mols[0], str):
            rdkit_mols = [MolFromSmiles(mol) for mol in dataset.mols]
        elif isinstance(dataset.mols[0], Mol):
            rdkit_mols = dataset.mols
        else:
            rdkit_mols = None

        # featurization process using DeepChem SmilesToSeq
        dataset.X = SmilesToSeq(
            char_to_idx=self.char_to_idx,
            max_len=self.max_len,
            pad_len=self.pad_len).featurize(rdkit_mols)

        # identify which rows did not get featurized
        indexes = []
        for i, feat in enumerate(dataset.X):
            if len(feat) == 0:
                indexes.append(i)
        # treat indexes with no featurization
        dataset.remove_elements(indexes)
        dataset.X = np.asarray([np.asarray(feat, dtype=object) for feat in dataset.X])
        return dataset
