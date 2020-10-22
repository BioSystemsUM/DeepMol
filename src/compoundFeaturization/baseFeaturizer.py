

#TODO work on this, __init__ ??? change functions etc, identations
from abc import ABC


class Featurizer(object):
  """Abstract class for calculating a set of features for a datapoint.
  """

  def featurize(self, datapoints, log_every_n=1000):
      #TODO review comments, change them
    """Calculate features for datapoints.
    Parameters
    ----------
    datapoints: Iterable[Any]
      A sequence of objects that you'd like to featurize. Subclassses of
      `Featurizer` should instantiate the `_featurize` method that featurizes
      objects in the sequence.
    log_every_n: int, default 1000
      Logs featurization progress every `log_every_n` steps.
    Returns
    -------
    np.ndarray
      A numpy array containing a featurized representation of `datapoints`.
    """
    datapoints = list(datapoints)
    features = []
    for i, point in enumerate(datapoints):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)
      try:
        features.append(self._featurize(point))
      except:
        logger.warning(
            "Failed to featurize datapoint %d. Appending empty array")
        features.append(np.array([]))

    features = np.asarray(features)
    return features

  def __call__(self, datapoints: Iterable[Any]):
    """Calculate features for datapoints.
    Parameters
    ----------
    datapoints: Iterable[Any]
      Any blob of data you like. Subclasss should instantiate this.
    """
    return self.featurize(datapoints)

  def _featurize(self, datapoint: Any):
    """Calculate features for a single datapoint.
    Parameters
    ----------
    datapoint: Any
      Any blob of data you like. Subclass should instantiate this.
    """
    raise NotImplementedError('Featurizer is not defined.')


class MolecularFeaturizer(Featurizer):
    """Abstract class for calculating a set of features for a
    molecule.
    The defining feature of a `MolecularFeaturizer` is that it
    uses SMILES strings and RDKit molecule objects to represent
    small molecules. All other featurizers which are subclasses of
    this class should plan to process input which comes as smiles
    strings or RDKit molecules.
    Child classes need to implement the _featurize method for
    calculating features for a single molecule.
    Notes
    -----
    The subclasses of this class require RDKit to be installed.
    """

    def featurize(self, molecules, log_every_n=1000):
    """Calculate features for molecules.
    Parameters
    ----------
    molecules: rdkit.Chem.rdchem.Mol / SMILES string / iterable
      RDKit Mol, or SMILES string or iterable sequence of RDKit mols/SMILES
      strings.
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.
    Returns
    -------
    features: np.ndarray
      A numpy array containing a featurized representation of `datapoints`.
    """
    try:
      from rdkit import Chem
      from rdkit.Chem import rdmolfiles
      from rdkit.Chem import rdmolops
      from rdkit.Chem.rdchem import Mol
    except ModuleNotFoundError:
      raise ValueError("This class requires RDKit to be installed.")

    # Special case handling of single molecule
    if isinstance(molecules, str) or isinstance(molecules, Mol):
      molecules = [molecules]
    else:
      # Convert iterables to list
      molecules = list(molecules)

    features = []
    for i, mol in enumerate(molecules):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)

      try:
        if isinstance(mol, str):
          # mol must be a RDKit Mol object, so parse a SMILES
          mol = Chem.MolFromSmiles(mol)
          # SMILES is unique, so set a canonical order of atoms
          new_order = rdmolfiles.CanonicalRankAtoms(mol)
          mol = rdmolops.RenumberAtoms(mol, new_order)

        features.append(self._featurize(mol))
      except Exception as e:
        if isinstance(mol, Chem.rdchem.Mol):
          mol = Chem.MolToSmiles(mol)
        logger.warning(
            "Failed to featurize datapoint %d, %s. Appending empty array", i,
            mol)
        logger.warning("Exception message: {}".format(e))
        features.append(np.array([]))

    features = np.asarray(features)
    return features