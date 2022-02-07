'''author: Bruno Pereira
date: 28/04/2021
'''

import numpy as np
from rdkit.Chem.rdchem import Mol

from Datasets.Datasets import Dataset
from deepchem.utils.conformers import ConformerGenerator
from deepchem.feat import RDKitDescriptors, SmilesToImage, SmilesToSeq, CoulombMatrix, CoulombMatrixEig, \
    ConvMolFeaturizer, WeaveFeaturizer, MolGraphConvFeaturizer, RawFeaturizer
from compoundFeaturization.baseFeaturizer import MolecularFeaturizer
from rdkit import Chem
from typing import List, Any, Optional, Dict, Iterable, Union


def find_maximum_number_atoms(molecules):
    '''Finds maximum number of atoms within a set of molecules'''
    best = 0
    for i, mol in enumerate(molecules):
        try:
            atoms = mol.GetNumAtoms()
            if atoms > best:
                best = atoms
        except Exception as e:
            print('Molecule with index', i, 'was not converted from SMILES into RDKIT object')
    return best


def get_conformers(molecules, generator):
    '''Gets conformers for molecules with a specific generator'''
    new_conformations = []
    for i, mol in enumerate(molecules):
        try:
            conf = generator.generate_conformers(mol)
            new_conformations.append(conf)
        except Exception as e:
            print('Molecules with index', i, 'was not able to achieve a correct conformation')
            print('Appending empty list')
            new_conformations.append([])
    return new_conformations


def get_dictionary_from_smiles(smiles, max_len):
    '''Dictionary of character to index mapping
    Adapted from deepchem'''

    pad_token = "<pad>"
    out_of_vocab_token = "<unk>"

    char_set = set()
    for smile in smiles:
        if len(smile) <= max_len:
            char_set.update(set(smile))

    unique_char_list = list(char_set) + [pad_token, out_of_vocab_token]
    dictionary = {letter: idx for idx, letter in enumerate(unique_char_list)}
    return dictionary


class ConvMolFeat(MolecularFeaturizer):
    """Duvenaud graph convolution, adapted from deepchem.
    Vector of descriptors for each atom in a molecule.
    The featurizers computes that vector of local descriptors.
    """

    def __init__(self, master_atom: bool = False, use_chirality: bool = False, atom_properties: Iterable[str] = [],
                 per_atom_fragmentation: bool = False):
        """
        Parameters
        ----------
        master_atom: Boolean
            if true create a fake atom with bonds to every other atom.
            Default to False.
        use_chirality: Boolean
            if true then make the resulting atom features aware of the chirality
            of the molecules in question. Default to False.
        atom_properties: list of string or None
            properties in the RDKit Mol object to use as additional atom-level features
            in the larger molecular feature. If None, then no atom-level properties
            are used. Default to None.
        per_atom_fragmentation: Boolean
            If True, then multiple "atom-depleted" versions of each molecule will be 
            created (using featurize() method). Default to False.
        """
        super().__init__()
        self.master_atom = master_atom
        self.use_chirality = use_chirality
        self.atom_properties = atom_properties
        self.per_atom_fragmentation = per_atom_fragmentation

    def _featurize(self, mol: Union[Mol, str], log_every_n=1000):
        # obtain new SMILE's strings

        if isinstance(mol, str):
            rdkit_mols = [Chem.MolFromSmiles(mol)]
        elif isinstance(mol, Mol):
            rdkit_mols = [mol]
        else:
            rdkit_mols = None

        # featurization process using DeepChem featurizers
        feature = ConvMolFeaturizer(
            master_atom=self.master_atom,
            use_chirality=self.use_chirality,
            atom_properties=self.atom_properties,
            per_atom_fragmentation=self.per_atom_fragmentation).featurize(rdkit_mols)

        assert feature[0].atom_features is not None

        return feature[0]


#    def _featurize(self, mol: Any)->np.ndarray:
#        try:
#            fp = ConvMolFeaturizer(master_atom = self.master_atom, use_chirality = self.use_chirality, atom_properties = self.atom_properties)._featurize(mol)
#        except Exception as e:
#            print('Error in smile: ' + str(mol))
#            fp = np.empty(1024, dtype = float)
#            fp[:] = np.NaN
#        fp = np.asarray(fp, dtype = object)
#        return fp

class WeaveFeat(MolecularFeaturizer):
    """Weave convolution featurization, adapted from deepchem.
    Require a quadratic matrix of interaction descriptors for each
    pair of atoms"""

    def __init__(self, graph_distance: bool = True, explicit_H: bool = False, use_chirality: bool = False,
                 max_pair_distance: Optional[int] = None):
        """
        Parameters
        ----------
        graph_distance: Boolean
            If True, use graph distance for distance features. Otherwise,
            use Euclidean distance. Molecules invoked must have valid conformer information
            if this option is set. Default to True.
        explicit_H: Boolean
            If true, model hydrogens in the molecule. Default to False.
        use_chirality: Boolean
            If true, use chiral information in the featurization. Default to False.
        max_pair_distance: Optional[int]
            This value can be a positive integer or None. This parameter determines the maximum
            graph distance at which pair features are computed. Default to None.
        """
        self.graph_distance = graph_distance
        self.explicit_H = explicit_H
        self.use_chirality = use_chirality
        self.max_pair_distance = max_pair_distance

    def featurize(self, dataset: Dataset, log_every_n=1000):
        # obtain new SMILE's strings
        print('Converting SMILES to Mol')
        if isinstance(dataset.mols[0], str):
            rdkit_mols = [Chem.MolFromSmiles(mol) for mol in dataset.mols]
        elif isinstance(dataset.mols[0], Mol):
            rdkit_mols = dataset.mols
        else:
            rdkit_mols = None

        # featurization process using DeepChem featurizers
        print('Featurizing datapoints')
        dataset.X = WeaveFeaturizer(
            graph_distance=self.graph_distance,
            explicit_H=self.explicit_H,
            use_chirality=self.use_chirality,
            max_pair_distance=self.max_pair_distance).featurize(rdkit_mols)

        # identify which rows did not get featurized
        indexes = []
        for i, feat in enumerate(dataset.X):
            if i % log_every_n == 0:
                print('Analyzing datapoint %i' % i)
            try:
                mol = feat.get_atom_features()
            except Exception as e:
                print('Failed to featurize datapoint %d, %s' % (i, dataset.mols[i]))
                indexes.append(i)
        # treat indexes with no featurization
        dataset.remove_elements(indexes)
        print('Elements with indexes: ', indexes, 'were removed due to lack of featurization.')

        return dataset


class MolGraphConvFeat(MolecularFeaturizer):
    """Featurizer of general graph convolution networks for molecules.
    Adapted from deepchem"""

    def __init__(self, use_edges: bool = False, use_chirality: bool = False, use_partial_charge: bool = False):
        """
        Parameters
        ----------
        use_edges: Boolean
            Whether to use edge features or not. Default to False.
        use_chirality: Boolean
            Whether to use chirality information or not. Default to False.
        use_partial_charge: Boolean
            Whether to use partial chrage data or not. If True, computes gasteiger
            charges. Default to False.
        """
        self.use_edges = use_edges
        self.use_chirality = use_chirality
        self.use_partial_charge = use_partial_charge


    def _featurize(self, mol: Union[Mol, str], log_every_n=1000):
        # obtain new SMILE's strings

        if isinstance(mol, str):
            rdkit_mols = [Chem.MolFromSmiles(mol)]
        elif isinstance(mol, Mol):
            rdkit_mols = [mol]
        else:
            rdkit_mols = None

        # featurization process using DeepChem featurizers
        feature = MolGraphConvFeaturizer(
            use_edges=self.use_edges,
            use_chirality=self.use_chirality,
            use_partial_charge=self.use_partial_charge).featurize(rdkit_mols)

        if feature[0].node_features is None:
            raise Exception

        return feature[0]


class CoulombFeat(MolecularFeaturizer):
    """Calculate coulomb matrices for molecules.
    Adapted from deepchem"""

    def __init__(self, max_atoms: int, remove_hydrogens: bool = False, randomize: bool = False, upper_tri: bool = False,
                 n_samples: Optional[int] = 1, seed: Optional[int] = None):
        """
        Parameters
        ----------
        max_atoms: int
            The maximum number of atoms expected for molecules this featurizers will
            process.
        remove_hydrogens: Boolean
            If True, remove hydrogens before processing them. Default to False.
        randomize: Boolean
            If True, randomize Coulomb matrices. Default to False.
        upper_tri: Boolean
            Generate only upper triangle part of Coulomb matrices. Default to False.
        n_samples: Optional[int]
            If 'randomize' is set to True, the number of random samples to draw.
            Default to 1.
        seed: Optional[int]
            Random seed to use. Default to None.
        """
        self.max_atoms = max_atoms
        self.remove_hydrogens = remove_hydrogens
        self.randomize = randomize
        self.upper_tri = upper_tri
        self.n_samples = n_samples
        if seed is not None:
            seed = int(seed)
        self.seed = seed

    def featurize(self, dataset: Dataset, max_conformers: int = 1, log_every_n=1000):
        # obtain new SMILE's strings
        print('Converting SMILES to Mol')
        if isinstance(dataset.mols[0], str):
            rdkit_mols = [Chem.MolFromSmiles(mol) for mol in dataset.mols]
        elif isinstance(dataset.mols[0], Mol):
            rdkit_mols = dataset.mols
        else:
            rdkit_mols = None

        generator = ConformerGenerator(max_conformers=max_conformers)

        # TO USE in case to add option for the software to find the parameter max_atoms
        # maximum_number_atoms = find_maximum_number_atoms(new_smiles)

        new_conformers = get_conformers(rdkit_mols, generator)
        # featurization process using DeepChem featurizers
        print('Featurizing datapoints')
        featurizer = CoulombMatrix(
            max_atoms=self.max_atoms,
            remove_hydrogens=self.remove_hydrogens,
            randomize=self.randomize,
            upper_tri=self.upper_tri,
            n_samples=self.n_samples,
            seed=self.seed)

        dataset.X = featurizer(new_conformers)

        # identify which rows did not get featurized
        indexes = []
        for i, feat in enumerate(dataset.X):
            if i % log_every_n == 0:
                print('Analyzing datapoint %i' % i)
            if feat.size == 0:
                print('Failed to featurize datapoint %d, %s' % (i, dataset.mols[i]))
                indexes.append(i)

        # treat indexes with no featurization
        dataset.remove_elements(indexes)
        print('Elements with indexes: ', indexes, 'were removed due to lack of featurization.')

        return dataset


class CoulombEigFeat(MolecularFeaturizer):
    """Calculate the eigen values of Coulomb matrices for molecules.
    Adapted from deepchem"""

    def __init__(self, max_atoms: int, remove_hydrogens: bool = False, randomize: bool = False,
                 n_samples: Optional[int] = 1, seed: Optional[int] = None):
        """
        Parameters
        ----------
        max_atoms: int
            The maximum number of atoms expected for molecules this featurizers will
            process.
        remove_hydrogens: Boolean
            If True, remove hydrogens before processing them. Default to False.
        randomize: Boolean
            If True, randomize Coulomb matrices. Default to False.
        n_samples: Optional[int]
            If 'randomize' is set to True, the number of random samples to draw.
            Default to 1.
        seed: Optional[int]
            Random seed to use. Default to None.
        """

        self.max_atoms = max_atoms
        self.remove_hydrogens = remove_hydrogens
        self.randomize = randomize
        self.n_samples = n_samples
        if seed is not None:
            seed = int(seed)
        self.seed = seed

    def featurize(self, dataset: Dataset, max_conformers: int = 1, log_every_n=1000):
        # obtain new SMILE's strings
        print('Converting SMILES to Mol')
        if isinstance(dataset.mols[0], str):
            rdkit_mols = [Chem.MolFromSmiles(mol) for mol in dataset.mols]
        elif isinstance(dataset.mols[0], Mol):
            rdkit_mols = dataset.mols
        else:
            rdkit_mols = None

        generator = ConformerGenerator(max_conformers=max_conformers)

        # TO USE in case to add option for the software to find the parameter max_atoms
        # maximum_number_atoms = find_maximum_number_atoms(new_smiles)

        new_conformers = get_conformers(rdkit_mols, generator)
        # featurization process using DeepChem featurizers
        print('Featurizing datapoints')
        featurizer = CoulombMatrixEig(
            max_atoms=self.max_atoms,
            remove_hydrogens=self.remove_hydrogens,
            randomize=self.randomize,
            n_samples=self.n_samples,
            seed=self.seed)

        dataset.X = featurizer(new_conformers)

        # identify which rows did not get featurized
        indexes = []
        for i, feat in enumerate(dataset.X):
            if i % log_every_n == 0:
                print('Analyzing datapoint %i' % i)
            if feat.size == 0:
                print('Failed to featurize datapoint %d, %s' % (i, dataset.mols[i]))
                indexes.append(i)

        # treat indexes with no featurization
        dataset.remove_elements(indexes)
        print('Elements with indexes: ', indexes, 'were removed due to lack of featurization.')

        return dataset


class SmileImageFeat(MolecularFeaturizer):
    """Converts SMILE string to image.
    Adapted from deepchem"""

    def __init__(self, img_size: int = 80, res: float = 0.5, max_len: int = 250, img_spec: str = "std"):
        """
        Parameters
        ----------
        img_size: int
            Size of the image tensor. Default to 80.
        res: float
            Displays the resolution of each pixel in Angstrom. Default to 0.5.
        max_len: int
            Maximum allowed length of SMILES string. Default to 250.
        img_spec: str
            Indicates the channel organization of the image tensor. Default to 'std'.
        """
        super().__init__()
        if img_spec not in ["std", "engd"]:
            raise ValueError(
                "Image mode must be one of the std or engd. {} is not supported".format(img_spec))
        self.img_size = img_size
        self.max_len = max_len
        self.res = res
        self.img_spec = img_spec
        self.embed = int(img_size * res / 2)

    #    def _featurize(self, mol: Any) -> np.ndarray:
    #        try:
    #            fp = SmilesToImage(
    #                    img_size = self.img_size,
    #                    max_len = self.max_len,
    #                    res = self.res,
    #                    img_spec = self.img_spec,
    #                   embed = self.embed)._featurize(mol)
    #        except Exception as e:
    #            print('Error in smile: ' + str(mol))
    #            fp = np.empty(self.img_size, dtype = float)
    #            fp[:] = np.NaN
    #        fp = np.asarray(fp, dtype = object)
    #        return fp

    def _featurize(self, mol: Union[Mol, str]):

        if isinstance(mol, str):
            rdkit_mols = [Chem.MolFromSmiles(mol)]
        elif isinstance(mol, Mol):
            rdkit_mols = [mol]
        else:
            rdkit_mols = None

        # featurization process using DeepChem featurizers
        feats = SmilesToImage(
            img_size=self.img_size,
            max_len=self.max_len,
            res=self.res,
            img_spec=self.img_spec).featurize(rdkit_mols)

        # identify which rows did not get featurized
        if len(feats[0]) == 0:
            raise Exception

        return feats


class SmilesSeqFeat(MolecularFeaturizer):
    """Takes SMILES strings and turns into a sequence.
    Adapated from deepchem"""

    def __init__(self, char_to_idx: Dict[str, int] = None, max_len: int = 250, pad_len: int = 10):
        """
        Parameters
        ----------
        char_to_idx: Dict
            Dictionary containing character to index mappings for unique characters.
            Default to None.
        max_len: int
            Maximum allowed length of the SMILES string. Default to 250.
        pad_len: int
            Amount of padding to add on either side of the SMILES seq. Default to 10
        """
        super().__init__()
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        self.pad_len = pad_len

    def _featurize(self, dataset: Dataset, log_every_n=1000):
        # Getting the dictionary if it is None
        if self.char_to_idx == None:

            if isinstance(dataset.mols[0], Mol):
                smiles = [Chem.MolToSmiles(mol) for mol in dataset.mols]
            elif isinstance(dataset.mols[0], str):
                smiles = dataset.mols
            else:
                smiles = None

            self.char_to_idx = get_dictionary_from_smiles(smiles, self.max_len)

        dataset.dictionary = self.char_to_idx

        # obtain new SMILE's strings
        print('Converting SMILES to Mol')
        if isinstance(dataset.mols[0], str):
            rdkit_mols = [Chem.MolFromSmiles(mol) for mol in dataset.mols]
        elif isinstance(dataset.mols[0], Mol):
            rdkit_mols = dataset.mols
        else:
            rdkit_mols = None

        # featurization process using DeepChem featurizers
        print('Featurizing datapoints')
        dataset.X = SmilesToSeq(
            char_to_idx=self.char_to_idx,
            max_len=self.max_len,
            pad_len=self.pad_len).featurize(rdkit_mols)

        # identify which rows did not get featurized
        indexes = []
        for i, feat in enumerate(dataset.X):
            if i % log_every_n == 0:
                print('Analyzing datapoint %i' % i)
            if len(feat) == 0:
                print('Failed to featurize datapoint %d, %s' % (i, dataset.mols[i]))
                indexes.append(i)
        # treat indexes with no featurization
        dataset.remove_elements(indexes)
        print('Elements with indexes: ', indexes, 'were removed due to lack of featurization.')
        dataset.X = np.asarray([np.asarray(feat, dtype=object) for feat in dataset.X])

        return dataset


class CGCNNFeat():
    """Calculate structure graph features for crystals.
    Adapted from deepchem. Not implemented (outside molecular domain)"""

    def __init__(self, radius: float = 8.0, max_neighbors: float = 12, step: float = 0.2):
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.step = step

    def _featurize(self, dataset: Dataset, log_every_n=1000):

        # Dataset is supposed to be composed by structure dictionaries or pymatgen.Structure objects
        try:
            from pymatgen import Structure
        except ModuleNotFoundError:
            raise ImportError("This class requires pymatgen to be installed")

        pass


class RawFeat(MolecularFeaturizer):

    def _featurize(self, mol: Union[Mol, str], log_every_n=1000):

        if isinstance(mol, Mol):
            smiles = Chem.MolToSmiles(mol)
        elif isinstance(mol, str):
            smiles = mol
        else:
            smiles = None

        mol = RawFeaturizer().featurize([smiles])[0]
        #dataset.ids = smiles  # this is needed when calling the build_char_dict method (TextCNNModel)
        return mol
