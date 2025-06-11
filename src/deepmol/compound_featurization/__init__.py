import warnings

from .base_featurizer import MolecularFeaturizer

from .rdkit_descriptors import ThreeDimensionalMoleculeGenerator, All3DDescriptors, AutoCorr3D, \
    RadialDistributionFunction, PlaneOfBestFit, MORSE, WHIM, RadiusOfGyration, InertialShapeFactor, Eccentricity, \
    Asphericity, SpherocityIndex, PrincipalMomentsOfInertia, NormalizedPrincipalMomentsRatios, \
    generate_conformers_to_sdf_file, TwoDimensionDescriptors

from .rdkit_fingerprints import MorganFingerprint, AtomPairFingerprint, LayeredFingerprint, RDKFingerprint, \
    MACCSkeysFingerprint

from .similarity_matrix import TanimotoSimilarityMatrix

from .mixed_descriptors import MixedFeaturizer

try:
    from .mol2vec import Mol2Vec
except ImportError:
    warnings.warn("Mol2Vec not available. Please install it to use it. "
                  "(pip install git+https://github.com/samoturk/mol2vec#egg=mol2vec)")

try:
    from .deepchem_featurizers import WeaveFeat, CoulombFeat, CoulombEigFeat, ConvMolFeat, MolGraphConvFeat, \
        SmileImageFeat, SmilesSeqFeat, MolGanFeat, PagtnMolGraphFeat, DagTransformer, DMPNNFeat, MATFeat, RawFeat
except ImportError:
    warnings.warn("DeepChem not available. Please install it to use it.")

from .one_hot_encoder import SmilesOneHotEncoder

from .np_classifier_fp import NPClassifierFP

from .nc_mfp_generator import NcMfp

try:
    from .neural_npfp_generator import NeuralNPFP
except ImportError:
    warnings.warn("Neural NPFP not available. Please install the 'deep-learning' or 'all' of deepmol to use it. "
                  "(pip install deepmol[deep-learning] or pip install deepmol[all])")

from .mhfp import MHFP

from .biosynfoni import BiosynfoniKeys

from .llms import LLM
