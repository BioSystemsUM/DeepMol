from typing import List, Dict, Any, Union

import numpy as np
from deepchem.data import NumpyDataset
from deepchem.feat import ConvMolFeaturizer, WeaveFeaturizer, MolGraphConvFeaturizer, CoulombMatrix, CoulombMatrixEig, \
    SmilesToImage, SmilesToSeq, MolGanFeaturizer, GraphMatrix, PagtnMolGraphFeaturizer, DMPNNFeaturizer, MATFeaturizer
from deepchem.feat.graph_data import GraphData
from deepchem.feat.mol_graphs import ConvMol, WeaveMol
from deepchem.feat.molecule_featurizers.mat_featurizer import MATEncoding
from deepchem.trans import DAGTransformer as DAGTransformerDC
from deepchem.utils import ConformerGenerator
from rdkit.Chem import Mol
from rdkit.Chem import rdFreeSASA

from deepmol.base import Transformer
from deepmol.compound_featurization import MolecularFeaturizer
from deepmol.compound_featurization._utils import get_conformers, get_dictionary_from_smiles
from deepmol.datasets import Dataset
from deepmol.loggers.logger import Logger
from deepmol.utils.decorators import modify_object_inplace_decorator
from deepmol.utils.utils import mol_to_smiles


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
                 atom_properties:bool = True,
                 per_atom_fragmentation: bool = False,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        master_atom: bool
            If True, create a fake atom with bonds to every other atom.
        use_chirality: bool
            If True, include chirality information.
        atom_properties: bool
            If True, include atom properties.
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

    def get_atom_features(self, mol):
        
        three_d_structure = False
        if mol.GetNumConformers() > 0:
            radii1 = rdFreeSASA.classifyAtoms(mol)
            rdFreeSASA.CalcSASA(mol, radii1)
            three_d_structure = True

        for atom in mol.GetAtoms():
            is_aromatic = atom.GetIsAromatic()
            explicit_valence = atom.GetExplicitValence()
            formal_charge = atom.GetFormalCharge()
            hybridization = atom.GetHybridization()
            implicit_valence = atom.GetImplicitValence()
            is_in_ring = atom.IsInRing()
            mass = atom.GetMass()
            num_explicit_hs = atom.GetNumExplicitHs()
            num_implicit_hs = atom.GetNumImplicitHs()
            atom_number = atom.GetAtomicNum()


            properties = [("is_aromatic", is_aromatic),
                                                ("explicit_valence", explicit_valence),
                                                ("formal_charge", formal_charge),
                                                ("hybridization", hybridization),
                                                ("implicit_valence", implicit_valence),
                                                ("is_in_ring", is_in_ring),
                                                ("mass", mass),
                                                ("num_explicit_hs", num_explicit_hs),
                                                ("num_implicit_hs", num_implicit_hs),
                                                ("atom_number", atom_number)]
            if three_d_structure:
                positions = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                x, y, z = positions.x, positions.y, positions.z
                sasa = float(atom.GetProp("SASA"))

                properties.extend([("x", x), ("y", y), ("z", z), ("sasa", sasa)])

            for property_name, property_value in properties:
                # Construct the property name
                atom_property_name = f"atom {atom.GetIdx():08d} {property_name}"
                
                # Assign the property to the molecule
                mol.SetDoubleProp(atom_property_name, property_value)
        return mol, properties
        

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
        if self.atom_properties:
            mol, properties = self.get_atom_features(mol)

            self.atom_properties_ = [name for name, _ in properties]
        else:
            self.atom_properties_ = []

        feature = ConvMolFeaturizer(
            master_atom=self.master_atom,
            use_chirality=self.use_chirality,
            atom_properties=self.atom_properties_,
            per_atom_fragmentation=self.per_atom_fragmentation).featurize([mol])

        assert feature[0].atom_features is not None
        return feature[0]


class PagtnMolGraphFeat(MolecularFeaturizer):
    """
    This class is a featurizer of PAGTN graph networks for molecules.

    The featurization is based on `PAGTN model <https://arxiv.org/abs/1905.12712>`_. It is slightly more computationally
    intensive than default Graph Convolution Featurizer, but it builds a Molecular Graph connecting all atom pairs
    accounting for interactions of an atom with every other atom in the Molecule. According to the paper, interactions
    between two pairs of atom are dependent on the relative distance between them and hence, the function needs
    to calculate the shortest path between them.

    References
    ----------
        [1] Chen, Barzilay, Jaakkola "Path-Augmented Graph Transformer Network"
        10.26434/chemrxiv.8214422.
    """

    def __init__(self,
                 max_length: int = 5,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        max_length: int
            Maximum distance up to which shortest paths must be considered.
            Paths shorter than max_length will be padded and longer will be
            truncated, default to ``5``.
        kwargs:
            Additional arguments for the base class.
        """
        super().__init__(**kwargs)
        self.max_length = max_length
        self.feature_names = ['pagtn_mol_graph_feat']

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
        # featurization process using DeepChem PagtnMolGraphFeaturizer
        feature = PagtnMolGraphFeaturizer(max_length=self.max_length).featurize([mol])

        assert feature[0].node_features is not None
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
                 generate_conformers: bool = True,
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
        generate_conformers: bool
            Whether to generate conformers.
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
        self.generate_conformers = generate_conformers
        self.feature_names = [f'coulomb_feat_{i}' for i in range(max_atoms)]

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
        if self.generate_conformers:
            generator = ConformerGenerator(max_conformers=self.max_conformers)
            new_conformers = get_conformers([mol], generator)
        else:
            new_conformers = [mol]

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
                 generate_conformers=True,
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
        generate_conformers: bool
            Whether to generate conformers.
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
        self.generate_conformers = generate_conformers
        self.feature_names = [f'coulomb_eig_feat_{i}' for i in range(max_atoms)]

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

        if self.generate_conformers:
            generator = ConformerGenerator(max_conformers=self.max_conformers)

            # TO USE in case to add option for the software to find the parameter max_atoms
            # maximum_number_atoms = find_maximum_number_atoms(new_smiles)

            new_conformers = get_conformers([mol], generator)
        else:
            new_conformers = [mol]
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
        self.feature_names = [f'smile_image_feat_{i}' for i in range(img_size)]

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


class SmilesSeqFeat(Transformer):
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
        super().__init__()
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        self.pad_len = pad_len
        self.feature_names = ['smiles_seq_feat']
        self.logger = Logger()

    @modify_object_inplace_decorator
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
                smiles = [mol_to_smiles(mol) for mol in dataset.mols if mol is not None]
            elif isinstance(dataset.mols[0], str):
                smiles = dataset.mols
            else:
                smiles = None

            self.char_to_idx = get_dictionary_from_smiles(smiles, self.max_len)

        dataset.dictionary = self.char_to_idx

        # obtain new SMILE's strings
        if isinstance(dataset.mols[0], str):
            rdkit_mols = [mol_to_smiles(mol) for mol in dataset.mols]
        elif isinstance(dataset.mols[0], Mol):
            rdkit_mols = dataset.mols
        else:
            rdkit_mols = None

        # featurization process using DeepChem SmilesToSeq
        dataset._X = SmilesToSeq(
            char_to_idx=self.char_to_idx,
            max_len=self.max_len,
            pad_len=self.pad_len).featurize(rdkit_mols)

        # identify which rows did not get featurized
        indexes = []
        for i, feat in enumerate(dataset._X):
            if len(feat) == 0:
                indexes.append(i)
        # treat indexes with no featurization
        dataset.remove_elements(indexes, inplace=True)
        dataset._X = np.asarray([np.asarray(feat, dtype=object) for feat in dataset._X if len(feat) > 0])
        return dataset

    def _fit(self, dataset: Dataset) -> 'SmilesSeqFeat':
        """
        Fit the featurizer. This method is called by the fit method of the Transformer class.
        To be used by pipeline.

        Parameters
        ----------
        dataset: Dataset
            The dataset containing the molecules to featurize in dataset.mols.

        Returns
        -------
        self: SmilesSeqFeat
            The fitted featurizer.
        """
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset. This method is called by the transform method of the Transformer class.
        To be used by pipeline.

        Parameters
        ----------
        dataset: Dataset
            The dataset containing the molecules to featurize in dataset.mols.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.
        """
        return self.featurize(dataset)


class DagTransformer(Transformer):
    """
    Performs transform from ConvMol adjacency lists to DAG calculation orders

    This transformer is used by `DAGModel` before training to transform its
    inputs to the correct shape. This expansion turns a molecule with `n` atoms
    into `n` DAGs, each with root at a different atom in the molecule.

    Reference:
    https://deepchem.readthedocs.io/en/latest/api_reference/transformers.html#dagtransformer
    """

    def __init__(self, max_atoms: int = 50) -> None:
        """
        Initialize this transformer.

        Parameters
        ----------
        max_atoms: int
            Maximum number of atoms in a molecule.
        """
        super().__init__()
        self.max_atoms = max_atoms
        self.feature_names = ['dag_transformer_feat']

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset. This method is called by the transform method of the Transformer class.
        Transforms ConvMolFeat features to DAG calculation orders.

        Parameters
        ----------
        dataset: Dataset
            Dataset to transform.

        Returns
        -------
        dataset: Dataset
            Transformed dataset.
        """
        if dataset.X is None:
            raise ValueError("Dataset must have X property set (ConvMolFeat).")
        dd = NumpyDataset(X=dataset._X, y=dataset.y, ids=dataset.ids, n_tasks=dataset.n_tasks)
        dd = DAGTransformerDC(max_atoms=self.max_atoms).transform(dd)
        dataset._X = dd.X
        dataset.feature_names = self.feature_names
        return dataset

    def _fit(self, dataset: Dataset) -> 'DagTransformer':
        """
        Fit the featurizer. This method is called by the fit method of the Transformer class.

        Parameters
        ----------
        dataset: Dataset
            The dataset containing the molecules to featurize in dataset.mols.

        Returns
        -------
        self: DagTransformer
            The fitted featurizer.
        """
        return self


class DMPNNFeat(MolecularFeaturizer):
    """
    Featurizes molecules using DeepChem DMPNNFeaturizer.

    This class is a featurizer for Directed Message Passing Neural Network (D-MPNN) implementation

    The default node(atom) and edge(bond) representations are based on
    `Analyzing Learned Molecular Representations for Property Prediction paper <https://arxiv.org/pdf/1904.01561.pdf>`_.

    Reference:
    https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#dmpnnfeaturizer
    """

    def __init__(self,
                 features_generators: List[str] = None,
                 is_adding_hs: bool = False,
                 use_original_atom_ranks: bool = False,
                 **kwargs) -> None:
        """
        Initialize this featurizer.

        Parameters
        ----------
        features_generators: List[str], default None
            List of global feature generators to be used.
        is_adding_hs: bool, default False
            Whether to add Hs or not.
        use_original_atom_ranks: bool, default False
            Whether to use original atom mapping or canonical atom mapping.
        kwargs: dict
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.features_generators = features_generators
        self.is_adding_hs = is_adding_hs
        self.use_original_atom_ranks = use_original_atom_ranks
        self.feature_names = ['dmpnn_feat']

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
            The DMPNN features of the molecule.
        """
        # featurization process using DeepChem DMPNNFeaturizer
        feature = DMPNNFeaturizer(features_generators=self.features_generators,
                                  is_adding_hs=self.is_adding_hs,
                                  use_original_atom_ranks=self.use_original_atom_ranks).featurize([mol])

        assert feature[0].node_features is not None

        return feature[0]


class MATFeat(MolecularFeaturizer):
    """
    Featurizes molecules using DeepChem MATFeaturizer.

    This class is a featurizer for Molecular Attribute Transformer (MAT) implementation
    The returned value is a numpy array which consists of molecular graph descriptions:
        - Node Features
        - Adjacency Matrix
        - Distance Matrix

    Reference:
    [1] https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#matfeaturizer
    [2] Lukasz Maziarka et al. “Molecule Attention Transformer`<https://arxiv.org/abs/2002.08264>`”
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize this featurizer.

        Parameters
        kwargs: dict
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.feature_names = ['mat_feat']

    def _featurize(self, mol: Mol) -> MATEncoding:
        """
        Featurizes a single molecule.

        Parameters
        ----------
        mol: Mol
            Molecule to featurize.

        Returns
        -------
        feature: GraphData
            The MATEncoding features of the molecule.
        """
        # featurization process using DeepChem DMPNNFeaturizer
        feature = MATFeaturizer().featurize([mol])

        assert feature[0].node_features is not None

        return feature[0]


class RawFeat(MolecularFeaturizer):

    def _featurize(self, mol: Union[Mol, str]):
        """
        A passthrough featurizer that returns the input molecule unchanged.

        Parameters
        ----------
        mol: Mol
            Molecule to featurize.

        Returns
        -------
        mol
        """

        return mol
