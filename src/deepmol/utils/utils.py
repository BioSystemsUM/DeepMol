import random

import pandas as pd
import joblib
import os
from typing import Any, cast, IO, List, Union, Tuple
import gzip
import pickle
import numpy as np

from rdkit.Chem import rdMolDescriptors, rdDepictor, Mol, RDKFingerprint, rdmolfiles, rdmolops
from rdkit.Chem import Draw
from IPython.display import SVG

from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
import tempfile
from PIL import Image

from IPython.display import display

from deepmol.loggers import Logger


def smiles_to_mol(smiles: str, **kwargs) -> Union[Mol, None]:
    """
    Convert SMILES to RDKit molecule object.
    Parameters
    ----------
    smiles: str
        SMILES string to convert.
   kwargs:
           Keyword arguments for `rdkit.Chem.MolFromSmiles`.
    Returns
    -------
    Mol
        RDKit molecule object.
    """
    try:
        return Chem.MolFromSmiles(smiles, **kwargs)
    except TypeError:
        return None


def mol_to_smiles(mol: Mol, **kwargs) -> Union[str, None]:
    """
    Convert SMILES to RDKit molecule object.
    Parameters
    ----------
    mol: Mol
        RDKit molecule object to convert.
   kwargs:
           Keyword arguments for `rdkit.Chem.MolToSmiles`.
    Returns
    -------
    smiles: str
        SMILES string.
    """
    try:
        return Chem.MolToSmiles(mol, **kwargs)
    except TypeError:
        return None


def canonicalize_mol_object(mol_object: Mol) -> Mol:
    """
    Canonicalize a molecule object.

    Parameters
    ----------
    mol_object: Mol
        Molecule object to canonicalize.

    Returns
    -------
    Mol
        Canonicalized molecule object.
    """
    try:
        # SMILES is unique, so set a canonical order of atoms
        new_order = rdmolfiles.CanonicalRankAtoms(mol_object)
        mol_object = rdmolops.RenumberAtoms(mol_object, new_order)
    except Exception as e:
        mol_object = mol_object

    return mol_object


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


def load_from_disk(filename: str) -> Any:
    """
    Load object from file.

    Parameters
    ----------
    filename: str
        A filename you want to load.

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


def normalize_labels_shape(y_pred: Union[List, np.ndarray], n_tasks: int) -> np.ndarray:
    """
    Function to transform output from predict_proba (prob(0) prob(1)) to predict format (0 or 1).

    Parameters
    ----------
    y_pred: array
        array with predictions
    n_tasks: int
        number of tasks

    Returns
    -------
    labels
        Array of predictions in the format [0, 1, 0, ...]/[[0, 1, 0, ...], [0, 1, 1, ...], ...]
    """
    if n_tasks == 1:
        labels = _normalize_singletask_labels_shape(y_pred)
    else:
        if isinstance(y_pred, np.ndarray):
            if len(y_pred.shape) == 3:
                y_pred = np.array([np.array([j[1] for j in i]) for i in y_pred]).T
        labels = []
        for task in y_pred:
            labels.append(_normalize_singletask_labels_shape(task))
        labels = np.array(labels).T
    return labels


def _normalize_singletask_labels_shape(y_pred: Union[List, np.ndarray]) -> np.ndarray:
    """
    Function to transform output from predict_proba (prob(0) prob(1)) to predict format (0 or 1).

    Parameters
    ----------
    y_pred: array
        array with predictions

    Returns
    -------
    labels
        Array of predictions in the format [0, 1, 0, ...]/[[0, 1, 0, ...], [0, 1, 1, ...], ...]
    """
    labels = []
    # list of probabilities in the format [0.1, 0.9, 0.2, ...]
    if isinstance(y_pred[0], (np.floating, float)):
        return np.array(y_pred)
    # list of lists of probabilities in the format [[0.1], [0.2], ...]
    elif len(y_pred[0]) == 1:
        return np.array([i[0] for i in y_pred])
    # list of lists of probabilities in the format [[0.1, 0.9], [0.2, 0.8], ...]
    elif len(y_pred[0]) == 2:
        return np.array([i[1] for i in y_pred])
    elif len(y_pred[0]) > 2:
        return np.array([np.argmax(i) for i in y_pred])
    else:
        raise ValueError("Unknown format for y_pred!")


# DRAWING

# TODO: check this (two keys)


def prepare_mol(mol: Mol, kekulize: bool):
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


def mol_to_svg(mol: Mol, molSize: Tuple[int, int] = (450, 200), kekulize: bool = True, drawer: object = None, **kwargs):
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
    mc = prepare_mol(mol, kekulize)
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
    return mol_to_svg(mol, molSize=molSize, highlightAtoms=atomsToUse, highlightAtomColors={atomID: (0.3, 0.3, 1)})


def draw_morgan_bit_on_molecule(mol: Mol,
                                bit: int,
                                radius: int = 2,
                                nBits: int = 2048,
                                chiral: bool = False,
                                molSize: Tuple[int, int] = (450, 200)):
    """
    Draw a molecule with a Morgan fingerprint bit highlighted.

    Parameters
    ----------
    mol: Mol
        Molecule to draw.
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
    info = {}
    rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=info, useChirality=chiral)

    logger = Logger()

    if bit not in info.keys():
        logger.info('Bits ON: ', info.keys())
        raise ValueError('Bit is off! Select a on bit')

    logger.info('Bit %d with %d hits!' % (bit, len(info[bit])))

    aid, rad = info[bit][0]
    return getSubstructDepiction(mol, aid, rad, molSize=molSize)


###############################
##### RDK FINGERPRINTS #####
###############################

def draw_rdk_bits(mol: Mol, bits: int, minPath: int = 2, maxPath: int = 7, fpSize: int = 2048):
    """
    Draw a molecule with a RDK fingerprint bit highlighted.

    Parameters
    ----------
    mol: Mol
        Molecule to draw.
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
    rdkbit = {}
    fp = RDKFingerprint(mol, minPath=minPath, maxPath=maxPath, fpSize=fpSize, bitInfo=rdkbit)
    logger = Logger()

    if isinstance(bits, int):
        if bits not in rdkbit.keys():
            logger.info(f'Bits ON: {rdkbit.keys()}')
            raise ValueError('Bit is off! Select a on bit')
        return Draw.DrawRDKitBit(mol, bits, rdkbit)

    elif isinstance(bits, list):
        bits_on = []
        for b in bits:
            if b in rdkbit.keys():
                bits_on.append(b)
            else:
                logger.info('Bit %d is off!' % (b))
        if len(bits_on) == 0:
            raise ValueError('All the selected bits are off! Select on bits!')
        elif len(bits_on) != len(bits):
            logger.info(f'Bits ON: {bits_on}')
        tpls = [(mol, x, rdkbit) for x in bits_on]
        return Draw.DrawRDKitBits(tpls, molsPerRow=5, legends=['bit_' + str(x) for x in bits_on])

    elif bits == 'ON':
        tpls = [(mol, x, rdkbit) for x in fp.GetOnBits()]
        return Draw.DrawRDKitBits(tpls, molsPerRow=5, legends=[str(x) for x in fp.GetOnBits()])

    else:
        raise ValueError('Bits must be intenger, list of integers or ON!')


def draw_rdk_bit_on_molecule(mol: Mol,
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
    mol: Mol
        Molecule to draw.
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
    logger = Logger()

    info = {}
    RDKFingerprint(mol, minPath=minPath, maxPath=maxPath, fpSize=fpSize, bitInfo=info)

    if bit not in info.keys():
        logger.info(f'Bits ON: {info.keys()}')
        raise ValueError('Bit is off! Select a on bit')

    logger.info('Bit %d with %d hits!' % (bit, len(info[bit])))

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
