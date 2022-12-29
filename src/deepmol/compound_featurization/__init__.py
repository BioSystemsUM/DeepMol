import warnings

from .base_featurizer import MolecularFeaturizer

from .rdkit_descriptors import ThreeDimensionalMoleculeGenerator, All3DDescriptors, AutoCorr3D, \
    RadialDistributionFunction, PlaneOfBestFit, MORSE, WHIM, RadiusOfGyration, InertialShapeFactor, Eccentricity, \
    Asphericity, SpherocityIndex, PrincipalMomentsOfInertia, NormalizedPrincipalMomentsRatios, \
    generate_conformers_to_sdf_file, TwoDimensionDescriptors

from .rdkit_fingerprints import MorganFingerprint, AtomPairFingerprint, LayeredFingerprint, RDKFingerprint, \
    MACCSkeysFingerprint

try:
    from .mol2vec import Mol2Vec
except ImportError:
    warnings.warn("Mol2Vec not available. Please install it to use it.")

from .deepchem_featurizers import WeaveFeat, CoulombFeat, CoulombEigFeat, RawFeat, ConvMolFeat, MolGraphConvFeat, \
    CoulombEigFeat, SmileImageFeat, SmilesSeqFeat

from .mixed_descriptors import MixedFeaturizer
