import random

import pandas as pd
import joblib
import os
from typing import Any, cast, IO, List, Union, Tuple
import gzip
import pickle
import numpy as np

from deepchem.trans import DAGTransformer, IRVTransformer
from deepchem.data import NumpyDataset
from deepmol.datasets import Dataset

from rdkit.Chem import rdMolDescriptors, rdDepictor, Mol, RDKFingerprint
from rdkit.Chem import Draw
from IPython.display import SVG

from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
import tempfile
from PIL import Image

from IPython.display import display


def load_pickle_file(input_file: str) -> Any:
    """
    Load from single, possibly gzipped, pickle file.

    Parameters
    ----------
    input_file: str
        The filename of pickle file. This function can load from gzipped pickle file like `XXXX.pkl.gz`.

    Returns
    -------
    Any
      The object which is loaded from the pickle file.
    """
    if ".gz" in input_file:
        with gzip.open(input_file, "rb") as unzipped_file:
            return pickle.load(cast(IO[bytes], unzipped_file))
    else:
        with open(input_file, "rb") as opened_file:
            return pickle.load(opened_file)


def save_to_disk(dataset: Union[np.ndarray, Dataset], filename: str, compress: int = 3):
    """
    Save a dataset to file.

    Parameters
    ----------
    dataset: Union[np.ndarray, Dataset]
        A dataset you want to save.
    filename: str
        Path to save data.
    compress: int, default 3
        The compress option when dumping joblib file.
  """
    if filename.endswith('.joblib'):
        joblib.dump(dataset, filename, compress=compress)
    elif filename.endswith('.npy'):
        np.save(filename, dataset)
    else:
        raise ValueError("Filename with unsupported extension: %s" % filename)


def load_from_disk(filename: str) -> Any:
    """
    Load a dataset from file.

    Parameters
    ----------
    filename: str
        A filename you want to load data.

    Returns
    -------
    Any
      A loaded object from file.
    """
    name = filename
    if os.path.splitext(name)[1] == ".gz":
        name = os.path.splitext(name)[0]
    extension = os.path.splitext(name)[1]
    if extension == ".pkl":
        return load_pickle_file(filename)
    elif extension == ".joblib":
        return joblib.load(filename)
    elif extension == ".csv":
        # First line of user-specified CSV *must* be header.
        df = pd.read_csv(filename, header=0)
        df = df.replace(np.nan, str(""), regex=True)
        return df
    elif extension == ".npy":
        return np.load(filename, allow_pickle=True)
    else:
        raise ValueError("Unrecognized filetype for %s" % filename)


def normalize_labels_shape(y_pred: Union[List, np.ndarray]):
    """
    Function to transform output from predict_proba (prob(0) prob(1)) to predict format (0 or 1).

    Parameters
    ----------
    y_pred: array
        array with predictions

    Returns
    -------
    labels
        Array of predictions in the predict format (0 or 1).
    """
    labels = []
    for i in y_pred:
        if isinstance(i, (np.floating, float)):
            labels.append(int(round(i)))
        elif len(i) == 2:
            if i[0] > i[1]:
                labels.append(0)
            else:
                labels.append(1)
        elif len(i) == 1:
            print(i)
            labels.append(int(round(i[0])))
    return np.array(labels)


def dag_transformation(dataset: Dataset, max_atoms: int = 10):
    """
    Function to transform ConvMol adjacency lists to DAG calculation orders.
    Adapted from deepchem

    Parameters
    ----------
    dataset: Dataset
        Dataset to transform.
    max_atoms: int
        Maximum number of atoms to allow.

    Returns
    -------
    dataset: Dataset
        Transformed dataset.
    """
    new_dataset = NumpyDataset(
        X=dataset.X,
        y=dataset.y,
        ids=dataset.mols)

    transformer = DAGTransformer(max_atoms=max_atoms)
    res = transformer.transform(new_dataset)
    dataset.mols = res.ids
    dataset.X = res.X
    dataset.y = res.y

    return dataset


def irv_transformation(dataset: Dataset, K: int = 10, n_tasks: int = 1):
    """
    Function to transfrom ECFP to IRV features, used by MultitaskIRVClassifier as preprocessing step
    Adapted from deepchem

    Parameters
    ----------
    dataset: Dataset
        Dataset to transform.
    K: int
        Number of IRV features to generate.
    n_tasks: int
        Number of tasks.

    Returns
    -------
    dataset: Dataset
        Transformed dataset.
    """
    try:
        dummy_y = dataset.y[:, n_tasks]
    except IndexError:
        dataset.y = np.reshape(dataset.y, (np.shape(dataset.y)[0], n_tasks))
    new_dataset = NumpyDataset(
        X=dataset.X,
        y=dataset.y,
        ids=dataset.mols)

    transformer = IRVTransformer(K, n_tasks, new_dataset)
    res = transformer.transform(new_dataset)
    dataset.mols = res.ids
    dataset.X = res.X
    dataset.y = np.reshape(res.y, (np.shape(res.y)[0],))

    return dataset


# DRAWING

# TODO: check this (two keys)
MACCSsmartsPatts = {
    1: ('?', 0),  # ISOTOPE
    2: ('[#104,#105,#106,#107,#106,#109,#110,#111,#112]', 0),  # atomic num >103 Not complete
    2: ('[#104]', 0),  # limit the above def'n since the RDKit only accepts up to #104
    3: ('[#32,#33,#34,#50,#51,#52,#82,#83,#84]', 0),  # Group IVa,Va,VIa Rows 4-6
    4: ('[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]', 0),  # actinide
    5: ('[Sc,Ti,Y,Zr,Hf]', 0),  # Group IIIB,IVB (Sc...)
    6: ('[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]', 0),  # Lanthanide
    7: ('[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]', 0),  # Group VB,VIB,VIIB
    8: ('[!#6;!#1]1~*~*~*~1', 0),  # QAAA@1
    9: ('[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]', 0),  # Group VIII (Fe...)
    10: ('[Be,Mg,Ca,Sr,Ba,Ra]', 0),  # Group IIa (Alkaline earth)
    11: ('*1~*~*~*~1', 0),  # 4M Ring
    12: ('[Cu,Zn,Ag,Cd,Au,Hg]', 0),  # Group IB,IIB (Cu..)
    13: ('[#8]~[#7](~[#6])~[#6]', 0),  # ON(C)C
    14: ('[#16]-[#16]', 0),  # S-S
    15: ('[#8]~[#6](~[#8])~[#8]', 0),  # OC(O)O
    16: ('[!#6;!#1]1~*~*~1', 0),  # QAA@1
    17: ('[#6]#[#6]', 0),  # CTC
    18: ('[#5,#13,#31,#49,#81]', 0),  # Group IIIA (B...)
    19: ('*1~*~*~*~*~*~*~1', 0),  # 7M Ring
    20: ('[#14]', 0),  # Si
    21: ('[#6]=[#6](~[!#6;!#1])~[!#6;!#1]', 0),  # C=C(Q)Q
    22: ('*1~*~*~1', 0),  # 3M Ring
    23: ('[#7]~[#6](~[#8])~[#8]', 0),  # NC(O)O
    24: ('[#7]-[#8]', 0),  # N-O
    25: ('[#7]~[#6](~[#7])~[#7]', 0),  # NC(N)N
    26: ('[#6]=;@[#6](@*)@*', 0),  # C$=C($A)$A
    27: ('[I]', 0),  # I
    28: ('[!#6;!#1]~[CH2]~[!#6;!#1]', 0),  # QCH2Q
    29: ('[#15]', 0),  # P
    30: ('[#6]~[!#6;!#1](~[#6])(~[#6])~*', 0),  # CQ(C)(C)A
    31: ('[!#6;!#1]~[F,Cl,Br,I]', 0),  # QX
    32: ('[#6]~[#16]~[#7]', 0),  # CSN
    33: ('[#7]~[#16]', 0),  # NS
    34: ('[CH2]=*', 0),  # CH2=A
    35: ('[Li,Na,K,Rb,Cs,Fr]', 0),  # Group IA (Alkali Metal)
    36: ('[#16R]', 0),  # S Heterocycle
    37: ('[#7]~[#6](~[#8])~[#7]', 0),  # NC(O)N
    38: ('[#7]~[#6](~[#6])~[#7]', 0),  # NC(C)N
    39: ('[#8]~[#16](~[#8])~[#8]', 0),  # OS(O)O
    40: ('[#16]-[#8]', 0),  # S-O
    41: ('[#6]#[#7]', 0),  # CTN
    42: ('F', 0),  # F
    43: ('[!#6;!#1;!H0]~*~[!#6;!#1;!H0]', 0),  # QHAQH
    44: ('[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]', 0),  # OTHER
    45: ('[#6]=[#6]~[#7]', 0),  # C=CN
    46: ('Br', 0),  # BR
    47: ('[#16]~*~[#7]', 0),  # SAN
    48: ('[#8]~[!#6;!#1](~[#8])(~[#8])', 0),  # OQ(O)O
    49: ('[!+0]', 0),  # CHARGE
    50: ('[#6]=[#6](~[#6])~[#6]', 0),  # C=C(C)C
    51: ('[#6]~[#16]~[#8]', 0),  # CSO
    52: ('[#7]~[#7]', 0),  # NN
    53: ('[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]', 0),  # QHAAAQH
    54: ('[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]', 0),  # QHAAQH
    55: ('[#8]~[#16]~[#8]', 0),  # OSO
    56: ('[#8]~[#7](~[#8])~[#6]', 0),  # ON(O)C
    57: ('[#8R]', 0),  # O Heterocycle
    58: ('[!#6;!#1]~[#16]~[!#6;!#1]', 0),  # QSQ
    59: ('[#16]!:*:*', 0),  # Snot%A%A
    60: ('[#16]=[#8]', 0),  # S=O
    61: ('*~[#16](~*)~*', 0),  # AS(A)A
    62: ('*@*!@*@*', 0),  # A$!A$A
    63: ('[#7]=[#8]', 0),  # N=O
    64: ('*@*!@[#16]', 0),  # A$A!S
    65: ('c:n', 0),  # C%N
    66: ('[#6]~[#6](~[#6])(~[#6])~*', 0),  # CC(C)(C)A
    67: ('[!#6;!#1]~[#16]', 0),  # QS
    68: ('[!#6;!#1;!H0]~[!#6;!#1;!H0]', 0),  # QHQH (&...) SPEC Incomplete
    69: ('[!#6;!#1]~[!#6;!#1;!H0]', 0),  # QQH
    70: ('[!#6;!#1]~[#7]~[!#6;!#1]', 0),  # QNQ
    71: ('[#7]~[#8]', 0),  # NO
    72: ('[#8]~*~*~[#8]', 0),  # OAAO
    73: ('[#16]=*', 0),  # S=A
    74: ('[CH3]~*~[CH3]', 0),  # CH3ACH3
    75: ('*!@[#7]@*', 0),  # A!N$A
    76: ('[#6]=[#6](~*)~*', 0),  # C=C(A)A
    77: ('[#7]~*~[#7]', 0),  # NAN
    78: ('[#6]=[#7]', 0),  # C=N
    79: ('[#7]~*~*~[#7]', 0),  # NAAN
    80: ('[#7]~*~*~*~[#7]', 0),  # NAAAN
    81: ('[#16]~*(~*)~*', 0),  # SA(A)A
    82: ('*~[CH2]~[!#6;!#1;!H0]', 0),  # ACH2QH
    83: ('[!#6;!#1]1~*~*~*~*~1', 0),  # QAAAA@1
    84: ('[NH2]', 0),  # NH2
    85: ('[#6]~[#7](~[#6])~[#6]', 0),  # CN(C)C
    86: ('[C;H2,H3][!#6;!#1][C;H2,H3]', 0),  # CH2QCH2
    87: ('[F,Cl,Br,I]!@*@*', 0),  # X!A$A
    88: ('[#16]', 0),  # S
    89: ('[#8]~*~*~*~[#8]', 0),  # OAAAO
    90:
        ('[$([!#6;!#1;!H0]~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[CH2;R]1)]',
         0),  # QHAACH2A
    91:
        (
            '[$([!#6;!#1;!H0]~*~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[R]@['
            'CH2;R]1),$([!#6;!#1;!H0]~*~[R]1@[R]@[CH2;R]1)]',
            0),  # QHAAACH2A
    92: ('[#8]~[#6](~[#7])~[#6]', 0),  # OC(N)C
    93: ('[!#6;!#1]~[CH3]', 0),  # QCH3
    94: ('[!#6;!#1]~[#7]', 0),  # QN
    95: ('[#7]~*~*~[#8]', 0),  # NAAO
    96: ('*1~*~*~*~*~1', 0),  # 5 M ring
    97: ('[#7]~*~*~*~[#8]', 0),  # NAAAO
    98: ('[!#6;!#1]1~*~*~*~*~*~1', 0),  # QAAAAA@1
    99: ('[#6]=[#6]', 0),  # C=C
    100: ('*~[CH2]~[#7]', 0),  # ACH2N
    101:
        (
            '[$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@['
            'R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@['
            'R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@['
            'R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1)]',
            0),  # 8M Ring or larger. This only handles up to ring sizes of 14
    102: ('[!#6;!#1]~[#8]', 0),  # QO
    103: ('Cl', 0),  # CL
    104: ('[!#6;!#1;!H0]~*~[CH2]~*', 0),  # QHACH2A
    105: ('*@*(@*)@*', 0),  # A$A($A)$A
    106: ('[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]', 0),  # QA(Q)Q
    107: ('[F,Cl,Br,I]~*(~*)~*', 0),  # XA(A)A
    108: ('[CH3]~*~*~*~[CH2]~*', 0),  # CH3AAACH2A
    109: ('*~[CH2]~[#8]', 0),  # ACH2O
    110: ('[#7]~[#6]~[#8]', 0),  # NCO
    111: ('[#7]~*~[CH2]~*', 0),  # NACH2A
    112: ('*~*(~*)(~*)~*', 0),  # AA(A)(A)A
    113: ('[#8]!:*:*', 0),  # Onot%A%A
    114: ('[CH3]~[CH2]~*', 0),  # CH3CH2A
    115: ('[CH3]~*~[CH2]~*', 0),  # CH3ACH2A
    116: ('[$([CH3]~*~*~[CH2]~*),$([CH3]~*1~*~[CH2]1)]', 0),  # CH3AACH2A
    117: ('[#7]~*~[#8]', 0),  # NAO
    118: ('[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]', 1),  # ACH2CH2A > 1
    119: ('[#7]=*', 0),  # N=A
    120: ('[!#6;R]', 1),  # Heterocyclic atom > 1 (&...) Spec Incomplete
    121: ('[#7;R]', 0),  # N Heterocycle
    122: ('*~[#7](~*)~*', 0),  # AN(A)A
    123: ('[#8]~[#6]~[#8]', 0),  # OCO
    124: ('[!#6;!#1]~[!#6;!#1]', 0),  # QQ
    125: ('?', 0),  # Aromatic Ring > 1
    126: ('*!@[#8]!@*', 0),  # A!O!A
    127: ('*@*!@[#8]', 1),  # A$A!O > 1 (&...) Spec Incomplete
    128:
        (
            '[$(*~[CH2]~*~*~*~[CH2]~*),$([R]1@[CH2;R]@[R]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[R]@[CH2;R]1),'
            '$(*~[CH2]~*~[R]1@[R]@[CH2;R]1)]',
            0),  # ACH2AAACH2A
    129: ('[$(*~[CH2]~*~*~[CH2]~*),$([R]1@[CH2]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[CH2;R]1)]',
          0),  # ACH2AACH2A
    130: ('[!#6;!#1]~[!#6;!#1]', 1),  # QQ > 1 (&...)  Spec Incomplete
    131: ('[!#6;!#1;!H0]', 1),  # QH > 1
    132: ('[#8]~*~[CH2]~*', 0),  # OACH2A
    133: ('*@*!@[#7]', 0),  # A$A!N
    134: ('[F,Cl,Br,I]', 0),  # X (HALOGEN)
    135: ('[#7]!:*:*', 0),  # Nnot%A%A
    136: ('[#8]=*', 1),  # O=A>1
    137: ('[!C;!c;R]', 0),  # Heterocycle
    138: ('[!#6;!#1]~[CH2]~*', 1),  # QCH2A>1 (&...) Spec Incomplete
    139: ('[O;!H0]', 0),  # OH
    140: ('[#8]', 3),  # O > 3 (&...) Spec Incomplete
    141: ('[CH3]', 2),  # CH3 > 2  (&...) Spec Incomplete
    142: ('[#7]', 1),  # N > 1
    143: ('*@*!@[#8]', 0),  # A$A!O
    144: ('*!:*:*!:*', 0),  # Anot%A%Anot%A
    145: ('*1~*~*~*~*~*~1', 1),  # 6M ring > 1
    146: ('[#8]', 2),  # O > 2
    147: ('[$(*~[CH2]~[CH2]~*),$([R]1@[CH2;R]@[CH2;R]1)]', 0),  # ACH2CH2A
    148: ('*~[!#6;!#1](~*)~*', 0),  # AQ(A)A
    149: ('[C;H3,H4]', 1),  # CH3 > 1
    150: ('*!@*@*!@*', 0),  # A!A$A!A
    151: ('[#7;!H0]', 0),  # NH
    152: ('[#8]~[#6](~[#6])~[#6]', 0),  # OC(C)C
    153: ('[!#6;!#1]~[CH2]~*', 0),  # QCH2A
    154: ('[#6]=[#8]', 0),  # C=O
    155: ('*!@[CH2]!@*', 0),  # A!CH2!A
    156: ('[#7]~*(~*)~*', 0),  # NA(A)A
    157: ('[#6]-[#8]', 0),  # C-O
    158: ('[#6]-[#7]', 0),  # C-N
    159: ('[#8]', 1),  # O>1
    160: ('[C;H3,H4]', 0),  # CH3
    161: ('[#7]', 0),  # N
    162: ('a', 0),  # Aromatic
    163: ('*1~*~*~*~*~*~1', 0),  # 6M Ring
    164: ('[#8]', 0),  # O
    165: ('[R]', 0),  # Ring
    166: ('?', 0),  # Fragments  FIX: this can't be done in SMARTS
}


###############################
######### MACCS KEYS #########
###############################

def draw_MACCS_Pattern(smiles: str, smarts_patt_index: int, path: str = None):
    """
    Draw a molecule with a MACCS key highlighted.

    Parameters
    ----------
    smiles: str
        SMILES string of the molecule to draw.
    smarts_patt_index: int
        Index of the MACCS key to highlight.
    path: str
        Path to save the image to. If None, the image is not saved.

    Returns
    -------
    im: PIL.Image.Image
        Image of the molecule with the MACCS key highlighted.
    """
    mol = Chem.MolFromSmiles(smiles)
    smart = MACCSsmartsPatts[smarts_patt_index][0]
    patt = Chem.MolFromSmarts(smart)
    print('Mol: ', smiles)
    print('Pattern: ', smart)

    if mol.HasSubstructMatch(patt):
        print('Pattern found!')
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
        if path is None:
            with tempfile.TemporaryDirectory() as tmpdirname:
                d.WriteDrawingText(tmpdirname + 'mol.png')
                im = Image.open(tmpdirname + 'mol.png')
                return im
        else:
            d.WriteDrawingText(path)
            im = Image.open(path)
            return im
    else:
        print('Pattern does not match molecule!')


###############################
##### MORGAN FINGERPRINTS #####
###############################


def draw_morgan_bits(molecule: str, bits: Union[int, str, List[int]], radius: int = 2, nBits: int = 2048):
    """
    Draw a molecule with Morgan fingerprint bits highlighted.

    Parameters
    ----------
    molecule: str
        SMILES string of the molecule to draw.
    bits: Union[int, str, List[int]]
        Bit(s) to highlight.
        If 'ON', all bits that are set to 1 are highlighted.
    radius: int
        Radius of the Morgan fingerprint.
    nBits: int
        Number of bits in the Morgan fingerprint.

    Returns
    -------
    DrawMorganBits
        Object containing the image of the molecule with the Morgan fingerprint bits highlighted.
    """
    bi = {}

    mol = Chem.MolFromSmiles(molecule)

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=bi)

    if isinstance(bits, int):
        if bits not in bi.keys():
            print('Bits ON: ', bi.keys())
            raise ValueError('Bit is off! Select a on bit')
        return Draw.DrawMorganBit(mol, bits, bi)

    elif isinstance(bits, list):
        bits_on = []
        for b in bits:
            if b in bi.keys():
                bits_on.append(b)
            else:
                print('Bit %d is off!' % (b))
        if len(bits_on) == 0:
            raise ValueError('All the selected bits are off! Select on bits!')
        elif len(bits_on) != len(bits):
            print('Using only bits ON: ', bits_on)
        tpls = [(mol, x, bi) for x in bits_on]
        return Draw.DrawMorganBits(tpls, molsPerRow=5, legends=['bit_' + str(x) for x in bits_on])

    elif bits == 'ON':
        tpls = [(mol, x, bi) for x in fp.GetOnBits()]
        return Draw.DrawMorganBits(tpls, molsPerRow=5, legends=[str(x) for x in fp.GetOnBits()])

    else:
        raise ValueError('Bits must be intenger, list of integers or ON!')


def prepareMol(mol: Mol, kekulize: bool):
    """
    Prepare a molecule for drawing.

    Parameters
    ----------
    mol: Mol
        Molecule to prepare.
    kekulize: bool
        If True, the molecule is kekulized.

    Returns
    -------
    mc: Mol
        Prepared molecule.
    """
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    return mc


def moltosvg(mol: Mol, molSize: Tuple[int, int] = (450, 200), kekulize: bool = True, drawer: object = None, **kwargs):
    """
    Convert a molecule to SVG.

    Parameters
    ----------
    mol: Mol
        Molecule to convert.
    molSize: Tuple[int, int]
        Size of the molecule.
    kekulize: bool
        If True, the molecule is kekulized.
    drawer: object
        Object to draw the molecule.
    **kwargs:
        Additional arguments for the drawer.

    Returns
    -------
    SVG
        The molecule in SVG format.
    """
    mc = prepareMol(mol, kekulize)
    if drawer is None:
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc, **kwargs)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return SVG(svg.replace('svg:', ''))


def getSubstructDepiction(mol: Mol, atomID: int, radius: int, molSize: Tuple[int, int] = (450, 200)):
    """
    Get a depiction of a substructure.

    Parameters
    ----------
    mol: Mol
        Molecule to draw.
    atomID: int
        ID of the atom to highlight.
    radius: int
        Radius of the substructure.
    molSize: Tuple[int, int]
        Size of the molecule.

    Returns
    -------
    SVG
        The molecule in SVG format.
    """
    if radius > 0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomID)
        atomsToUse = []
        for b in env:
            atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        atomsToUse = list(set(atomsToUse))
    else:
        atomsToUse = [atomID]
    return moltosvg(mol, molSize=molSize, highlightAtoms=atomsToUse, highlightAtomColors={atomID: (0.3, 0.3, 1)})


def draw_morgan_bit_on_molecule(mol_smiles: str,
                                bit: int,
                                radius: int = 2,
                                nBits: int = 2048,
                                chiral: bool = False,
                                molSize: Tuple[int, int] = (450, 200)):
    """
    Draw a molecule with a Morgan fingerprint bit highlighted.

    Parameters
    ----------
    mol_smiles: str
        SMILES string of the molecule to draw.
    bit: int
        Bit to highlight.
    radius: int
        Radius of the Morgan fingerprint.
    nBits: int
        Number of bits in the Morgan fingerprint.
    chiral: bool
        If True, the molecule is drawn with chiral information.
    molSize: Tuple[int, int]
        Size of the molecule.

    Returns
    -------
    SVG
        The molecule in SVG format.
    """
    try:
        mol = Chem.MolFromSmiles(mol_smiles)
    except Exception as e:
        raise ValueError('Invalid SMILES.')

    info = {}
    rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=info, useChirality=chiral)

    if bit not in info.keys():
        print('Bits ON: ', info.keys())
        raise ValueError('Bit is off! Select a on bit')

    print('Bit %d with %d hits!' % (bit, len(info[bit])))

    aid, rad = info[bit][0]
    return getSubstructDepiction(mol, aid, rad, molSize=molSize)


###############################
##### RDK FINGERPRINTS #####
###############################

def draw_rdk_bits(smiles: str, bits: int, minPath: int = 2, maxPath: int = 7, fpSize: int = 2048):
    """
    Draw a molecule with a RDK fingerprint bit highlighted.

    Parameters
    ----------
    smiles: str
        SMILES string of the molecule to draw.
    bits: int
        Bit to highlight.
    minPath: int
        Minimum path length.
    maxPath: int
        Maximum path length.
    fpSize: int
        Number of bits in the fingerprint.

    Returns
    -------
    Draw.DrawRDKitBits
        The molecule with the fingerprint bits.
    """
    mol = Chem.MolFromSmiles(smiles)

    rdkbit = {}
    fp = RDKFingerprint(mol, minPath=minPath, maxPath=maxPath, fpSize=fpSize, bitInfo=rdkbit)

    if isinstance(bits, int):
        if bits not in rdkbit.keys():
            print('Bits ON: ', rdkbit.keys())
            raise ValueError('Bit is off! Select a on bit')
        return Draw.DrawRDKitBit(mol, bits, rdkbit)

    elif isinstance(bits, list):
        bits_on = []
        for b in bits:
            if b in rdkbit.keys():
                bits_on.append(b)
            else:
                print('Bit %d is off!' % (b))
        if len(bits_on) == 0:
            raise ValueError('All the selected bits are off! Select on bits!')
        elif len(bits_on) != len(bits):
            print('Using only bits ON: ', bits_on)
        tpls = [(mol, x, rdkbit) for x in bits_on]
        return Draw.DrawRDKitBits(tpls, molsPerRow=5, legends=['bit_' + str(x) for x in bits_on])

    elif bits == 'ON':
        tpls = [(mol, x, rdkbit) for x in fp.GetOnBits()]
        return Draw.DrawRDKitBits(tpls, molsPerRow=5, legends=[str(x) for x in fp.GetOnBits()])

    else:
        raise ValueError('Bits must be intenger, list of integers or ON!')


def draw_rdk_bit_on_molecule(mol_smiles: str,
                             bit: int,
                             minPath: int = 1,
                             maxPath: int = 7,
                             fpSize: int = 2048,
                             path_dir: str = None,
                             molSize: Tuple[int, int] = (450, 200)):
    """
    Draw a molecule with a RDK fingerprint bit highlighted.

    Parameters
    ----------
    mol_smiles: str
        SMILES string of the molecule to draw.
    bit: int
        Bit to highlight.
    minPath: int
        Minimum path length.
    maxPath: int
        Maximum path length.
    fpSize: int
        Number of bits in the fingerprint.
    path_dir: str
        Path to save the image.
    molSize: Tuple[int, int]
        Size of the molecule.

    Returns
    -------
    Images
        The molecule with the fingerprint bit highlighted.
    """
    try:
        mol = Chem.MolFromSmiles(mol_smiles)
    except Exception as e:
        raise ValueError('Invalid SMILES.')

    info = {}
    RDKFingerprint(mol, minPath=minPath, maxPath=maxPath, fpSize=fpSize, bitInfo=info)

    if bit not in info.keys():
        print('Bits ON: ', info.keys())
        raise ValueError('Bit is off! Select a on bit')

    print('Bit %d with %d hits!' % (bit, len(info[bit])))

    images = []
    for i in range(len(info[bit])):
        d = rdMolDraw2D.MolDraw2DCairo(molSize[0], molSize[1])
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightBonds=info[bit][i])
        d.FinishDrawing()
        if path_dir is None:
            with tempfile.TemporaryDirectory() as tmpdirname:
                d.WriteDrawingText(tmpdirname + 'mol_' + str(i) + '.png')
                im = Image.open(tmpdirname + 'mol_' + str(i) + '.png')
                images.append(im)
        else:
            d.WriteDrawingText(path_dir + 'mol_' + str(i) + '.png')
            im = Image.open(path_dir + 'mol_' + str(i) + '.png')
            images.append(im)
    return display(*images)
