import inspect
import sys
import traceback
from typing import Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol, AllChem, MolFromSmiles, Descriptors, rdMolDescriptors
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMoleculeConfs
from rdkit.ML.Descriptors import MoleculeDescriptors

from deepmol.compound_featurization import MolecularFeaturizer
from deepmol.datasets import Dataset
from deepmol.utils.errors import PreConditionViolationException


def _no_conformers_message(e):
    """
    Print a message when no conformers are found.

    Parameters
    ----------
    e : Exception
        Exception to print.
    """
    exc = traceback.format_exc()

    if isinstance(e, RuntimeError) and "molecule has no conformers" in exc \
            or isinstance(e, ValueError) and "Bad Conformer Id" in exc:
        print("You have to generate molecular conformers for each molecule. \n"
              "You can execute the following method: \n"
              "rdkit3DDescriptors.generate_conformers_to_sdf_file(dataset: Dataset, file_path: str,"
              " n_conformations: int,max_iterations: int, threads: int, timeout_per_molecule: int) \n"
              "The result will be stored in a SDF format file which can be loaded with the "
              "method: loaders.Loaders.SDFLoader()\n\n"
              "Or set the generate_conformers parameter to True")

        exit(1)


def check_atoms_coordinates(mol):
    """
    Function to check if a molecule contains zero coordinates in all atoms.
    Then this molecule must be eliminated.

    Example:
        # Load  test set to a frame
        sdf = 'miniset.sdf'
        df = pt.LoadSDF(sdf, molColName='mol3DProt')
        ## Checking if molecule contains only ZERO coordinates,
        ##  then remove that molecules from dataset
        df['check_coordinates'] = [checkAtomsCoordinates(x) for x in df.mol3DProt]
        df_eliminated_mols = dfl[df.check_coordinates == False]
        df = df[df.check_coordinates == True]
        df.drop(columns=['check_coordinates'], inplace=True)
        print('final minitest set:', df.shape[0])
        print('minitest eliminated:', df_eliminated_mols.shape[0])

    Parameters
    ----------
    mol : Mol
        Molecule to check coordinates.

    Returns
    -------
    bool
        True if molecule is OK and False if molecule contains zero coordinates.
    """
    try:
        conf = mol.GetConformer()
        position = []
        for i in range(conf.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            position.append([pos.x, pos.y, pos.z])
        position = np.array(position)
        if not np.any(position):
            return False
        else:
            return True
    except:
        return False


class ThreeDimensionalMoleculeGenerator:
    """
    Class to generate three-dimensional conformers and optimize them.
    """

    def __init__(self,
                 n_conformations: int = 5,
                 max_iterations: int = 5,
                 threads: int = 1,
                 timeout_per_molecule: int = 40):
        """
        Initialize the class.

        Parameters
        ----------
        n_conformations: int
          Number of conformations to be generated per molecule.
        max_iterations: int
          Maximum of iterations when optimizing molecular geometry.
        threads: int
          Number of threads to use.
        timeout_per_molecule: int
          Maximum time to be spent in each molecule.
        """
        self.max_iterations = max_iterations
        self.n_conformations = n_conformations
        self.threads = threads
        self.timeout_per_molecule = timeout_per_molecule

    @staticmethod
    def check_if_mol_has_explicit_hydrogens(new_mol: Mol):
        """
        Method to check if a molecule has explicit hydrogens.

        Parameters
        ----------
        new_mol: Mol
            Mol object from rdkit.

        Returns
        -------
        bool
            True if molecule has explicit hydrogens and False if not.
        """
        atoms = new_mol.GetAtoms()
        for atom in atoms:
            number_hydrogens = atom.GetNumExplicitHs()
            if number_hydrogens != 0:
                return True
        return False

    def generate_conformers(self, new_mol: Mol, etkdg_version: int = 1, **kwargs):
        """
        method to generate three-dimensional conformers

        Parameters
        ----------
        new_mol: Mol
          Mol object from rdkit
        etkdg_version: int
          version of the experimental-torsion-knowledge distance geometry (ETKDG) algorithm
        kwargs: dict
            Parameters for the ETKDG algorithm.

        Returns
        -------
        new_mol: Mol
            Mol object with three-dimensional conformers.
        """

        new_mol = Chem.AddHs(new_mol)

        if etkdg_version == 1:
            AllChem.EmbedMultipleConfs(new_mol, numConfs=self.n_conformations,
                                       params=AllChem.ETKDG(), **kwargs)

        elif etkdg_version == 2:
            AllChem.EmbedMultipleConfs(new_mol, numConfs=self.n_conformations,
                                       params=AllChem.ETKDGv2(), **kwargs)

        elif etkdg_version == 3:
            AllChem.EmbedMultipleConfs(new_mol, numConfs=self.n_conformations,
                                       params=AllChem.ETKDGv3(), **kwargs)

        else:
            print("Choose ETKDG's valid version (1,2 or 3)")
            return None

        return new_mol

    def optimize_molecular_geometry(self, mol: Mol, mode: str = "MMFF94"):
        """
        Class to generate three-dimensional conformers

        Parameters
        ----------
        mol: Mol
          Mol object from rdkit.
        mode: str
          mode for the molecular geometry optimization (MMFF or UFF variants).

        Returns
        -------
        mol: Mol
            Mol object with optimized molecular geometry.
        """

        mol = Chem.AddHs(mol)

        if "MMFF" in mode:
            AllChem.MMFFOptimizeMoleculeConfs(mol,
                                              maxIters=self.max_iterations,
                                              numThreads=self.threads,
                                              mmffVariant=mode)

        elif mode == "UFF":

            UFFOptimizeMoleculeConfs(mol,
                                     maxIters=self.max_iterations,
                                     numThreads=self.threads)

        return mol


def get_all_3D_descriptors(mol):
    """
    Method that lists all the methods and uses them to featurize the whole set.

    Parameters
    ----------
    mol: Mol
        Mol object from rdkit.

    Returns
    -------
    all_descriptors: list
        List with all the 3D descriptors.
    """
    size = 639
    current_module = sys.modules[__name__]

    all_descriptors = np.empty(0, dtype=float)
    for name, featurizer_function in inspect.getmembers(current_module, inspect.isclass):
        try:
            if issubclass(featurizer_function, ThreeDimensionDescriptor) and \
                    issubclass(featurizer_function, MolecularFeaturizer) and \
                    name not in [All3DDescriptors.__name__, ThreeDimensionDescriptor.__name__]:

                descriptor_function = featurizer_function(False)
                descriptor_values = descriptor_function._featurize(mol)

                if not np.any(np.isnan(descriptor_values)):
                    all_descriptors = np.concatenate((all_descriptors, descriptor_values))
                else:
                    raise Exception

        except Exception:
            print('error in molecule: ' + str(mol))
            all_descriptors = np.empty(size, dtype=float)
            all_descriptors[:] = np.NaN
            break
    return all_descriptors


def generate_conformers(generator: ThreeDimensionalMoleculeGenerator,
                        new_mol: Union[Mol, str],
                        etkg_version: int = 1,
                        optimization_mode: str = "MMFF94"):
    """
    Method to generate three-dimensional conformers and optimize them.

    Parameters
    ----------
    generator: ThreeDimensionalMoleculeGenerator
        Class to generate three-dimensional conformers and optimize them.
    new_mol: Union[Mol, str]
        Mol object from rdkit or SMILES string to generate conformers and optimize them.
    etkg_version: int
        version of the experimental-torsion-knowledge distance geometry (ETKDG) algorithm.
    optimization_mode: str
        mode for the molecular geometry optimization (MMFF or UFF variants).

    Returns
    -------
    new_mol: Mol
        Mol object with three-dimensional conformers and optimized molecular geometry.
    """
    if isinstance(new_mol, str):
        new_mol = MolFromSmiles(new_mol)

    new_mol = generator.generate_conformers(new_mol, etkg_version)
    new_mol = generator.optimize_molecular_geometry(new_mol, optimization_mode)
    return new_mol


# TODO : check whether sdf file is being correctly exported for multi-class classification
def generate_conformers_to_sdf_file(dataset: Dataset,
                                    file_path: str,
                                    n_conformations: int = 20,
                                    max_iterations: int = 5,
                                    threads: int = 1,
                                    timeout_per_molecule: int = 12,
                                    etkg_version: int = 1,
                                    optimization_mode: str = "MMFF94"):
    """
    Generate conformers using the experimental-torsion-knowledge distance geometry (ETKDG) algorithm from RDKit,
    optimize them and save in an SDF file.

    Parameters
    ----------
    dataset: Dataset
        DeepMol Dataset object
    file_path: str
        file_path where the conformers will be saved.
    n_conformations: int
        The number of conformations per molecule.
    max_iterations: int
        Maximum number of iterations for the molecule's conformers optimization.
    threads: int
        Number of threads.
    timeout_per_molecule: int
        The number of seconds in which the conformers are to be generated.
    etkg_version: int
        Version of the experimental-torsion-knowledge distance geometry (ETKDG) algorithm.
    optimization_mode: str
        Mode for the molecular geometry optimization (MMFF or UFF).
    """

    def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='|'):
        """
        Call in a loop to create terminal progress bar.

        Parameters
        ----------
        iteration: int
            Current iteration.
        total: int
            Total iterations.
        prefix: str
            Prefix string.
        suffix: str
            Suffix string.
        decimals: int
            Positive number of decimals in percent complete.
        length: int
            Character length of bar.
        fill: str
            Bar fill character.
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end="", flush=True)
        # Print New Line on Complete
        if iteration == total:
            print()

    generator = ThreeDimensionalMoleculeGenerator(max_iterations, n_conformations, threads, timeout_per_molecule)
    mol_set = dataset.mols
    final_set_with_conformations = []
    writer = Chem.SDWriter(file_path)

    for i in range(mol_set.shape[0]):
        printProgressBar(i, mol_set.shape[0])
        try:
            m2 = generate_conformers(generator, mol_set[i], etkg_version, optimization_mode)
            label = dataset.y[i]
            m2.SetProp("_Class", "%f" % label)
            if dataset.ids is not None and dataset.ids.size > 0:
                mol_id = dataset.ids[i]
                m2.SetProp("_ID", "%f" % mol_id)
            writer.write(m2)
            final_set_with_conformations.append(m2)
        except:
            pass

    writer.close()


class TwoDimensionDescriptors(MolecularFeaturizer):
    """
    Class to generate two-dimensional descriptors.
    It generates all descriptors from the RDKit library.
    """

    def __init__(self):
        """
        Initialize the class.
        """
        super().__init__()

    def _featurize(self, mol: Mol):
        """
        Generate all descriptors from the RDKit library.

        Parameters
        ----------
        mol: Mol
            Mol object from rdkit.

        Returns
        -------
        all_descriptors: np.ndarray
            Array with all 2D descriptors from rdkit.
        """
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        # header = calc.GetDescriptorNames()

        try:
            descriptors = calc.CalcDescriptors(mol)
            if np.isnan(np.sum(descriptors)):
                raise Exception
        except Exception as e:

            print('error in smile: ' + str(mol))
            _no_conformers_message(e)

            descriptors = np.empty(208, dtype=np.float64)
            descriptors[:] = np.NaN

        descriptors = np.array(descriptors, dtype=np.float64)
        return descriptors


class ThreeDimensionDescriptor(MolecularFeaturizer):
    """
    Class to generate three-dimensional descriptors.
    """

    def __init__(self, mandatory_generation_of_conformers):
        """
        Initialize the class.

        Parameters
        ----------
        mandatory_generation_of_conformers: bool
            If True, the conformers are generated and optimized before the descriptors are calculated.
        """
        self.descriptor_function = None
        self.mandatory_generation_of_conformers = mandatory_generation_of_conformers
        if self.mandatory_generation_of_conformers:
            self.three_dimensional_generator = ThreeDimensionalMoleculeGenerator()
        super().__init__()

    @property
    def descriptor_function(self):
        """
        Get the descriptor function.
        """
        return self._descriptor_function

    @descriptor_function.setter
    def descriptor_function(self, function: callable):
        """
        Set the descriptor function.

        Parameters
        ----------
        function: callable
            Function to calculate the descriptors.
        """
        self._descriptor_function = function

    def generate_descriptor(self, mol):
        """
        Generate the descriptors.

        Parameters
        ----------
        mol: Mol
            Mol object from rdkit.

        Returns
        -------
        descriptors: np.ndarray
            Array with the descriptors.
        """
        try:
            has_conformers = check_atoms_coordinates(mol)

            if not has_conformers and self.mandatory_generation_of_conformers:
                mol = self.three_dimensional_generator.generate_conformers(mol)
                mol = self.three_dimensional_generator.optimize_molecular_geometry(mol)
            elif not has_conformers:
                raise PreConditionViolationException("molecule has no conformers")

            fp = self.descriptor_function(mol)
            if any([isinstance(fp, fp_type) for fp_type in [str, int, np.float64, float, np.int64]]):
                fp = [fp]

        except PreConditionViolationException as e:
            _no_conformers_message(e)
            raise e

        except Exception as e:
            print('error in smile: ' + str(mol))
            _no_conformers_message(e)

            fp = np.empty(80, dtype=float)
            fp[:] = np.NaN

        fp = np.asarray(fp, dtype=float)
        return fp

    def _featurize(self, mol: Mol):
        raise NotImplementedError


class All3DDescriptors(MolecularFeaturizer):
    """
    Class to generate all three-dimensional descriptors.
    """

    def __init__(self, mandatory_generation_of_conformers=True):
        """
        Initialize the class.

        Parameters
        ----------
        mandatory_generation_of_conformers: bool
            If True, the conformers are generated and optimized before the descriptors are calculated.
        """
        self.generate_conformers = mandatory_generation_of_conformers
        if self.generate_conformers:
            self.three_dimensional_generator = ThreeDimensionalMoleculeGenerator()

        super().__init__()

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Featurization of a molecule with all rdkit 3D descriptors

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
          A numpy array of all 3D descriptors from rdkit.
        """

        size = 639
        try:
            has_conformers = check_atoms_coordinates(mol)

            if not has_conformers and self.generate_conformers:
                mol = self.three_dimensional_generator.generate_conformers(mol)
                mol = self.three_dimensional_generator.optimize_molecular_geometry(mol)
            elif not has_conformers:
                raise PreConditionViolationException("molecule has no conformers")

            fp = get_all_3D_descriptors(mol)

        except PreConditionViolationException as e:
            _no_conformers_message(e)
            raise e

        except Exception as e:
            print('error in smile: ' + str(mol))
            fp = np.empty(size, dtype=float)
            fp[:] = np.NaN

        fp = np.asarray(fp, dtype=float)
        return fp


class AutoCorr3D(ThreeDimensionDescriptor):
    """
    AutoCorr3D.
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        """
        Initialize the class.

        Parameters
        ----------
        mandatory_generation_of_conformers: bool
            If True, the conformers are generated and optimized before the descriptors are calculated.
        """
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcAUTOCORR3D

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        3D Autocorrelation descriptors vector calculation (length of 80)

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
          A numpy array of 3D Autocorrelation descriptors
        """
        fp = self.generate_descriptor(mol)
        return fp


class RadialDistributionFunction(ThreeDimensionDescriptor):
    """
    Radial distribution function
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        """
        Initialize the class.

        Parameters
        ----------
        mandatory_generation_of_conformers: bool
            If True, the conformers are generated and optimized before the descriptors are calculated.
        """
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcRDF

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Radial distribution function descriptors calculation (length of 210).

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
          A numpy array of Radial distribution function results.
        """
        fp = self.generate_descriptor(mol)
        return fp


class PlaneOfBestFit(ThreeDimensionDescriptor):
    """
    Plane of best fit
    Nicholas C. Firth, Nathan Brown, and Julian Blagg, JCIM 52:2516-25
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        """
        Initialize the class.

        Parameters
        ----------
        mandatory_generation_of_conformers: bool
            If True, the conformers are generated and optimized before the descriptors are calculated.
        """
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcPBF

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Radial distribution function descriptors calculation.

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
          A numpy array with the Plane of best fit.
        """
        fp = self.generate_descriptor(mol)
        return fp


class MORSE(ThreeDimensionDescriptor):
    """
    Molecule Representation of Structures based on Electron diffraction descriptors
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        """
        Initialize the class.

        Parameters
        ----------
        mandatory_generation_of_conformers: bool
            If True, the conformers are generated and optimized before the descriptors are calculated.
        """
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcMORSE

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Molecule Representation of Structures based on Electron diffraction descriptors calculation (length of 224).

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
          A numpy array of MORSE descriptors
        """
        fp = self.generate_descriptor(mol)
        return fp


class WHIM(ThreeDimensionDescriptor):
    """
    WHIM descriptors vector
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        """
        Initialize the class.

        Parameters
        ----------
        mandatory_generation_of_conformers: bool
            If True, the conformers are generated and optimized before the descriptors are calculated.
        """
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcWHIM

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        WHIM descriptors calculation (length of 114).

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
          A numpy array of WHIM descriptors
        """
        fp = self.generate_descriptor(mol)
        return fp


class RadiusOfGyration(ThreeDimensionDescriptor):
    """
    Calculate Radius of Gyration
    G. A. Arteca “Molecular Shape Descriptors” Reviews in Computational Chemistry vol 9
    https://doi.org/10.1002/9780470125861.ch5
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        """
        Initialize the class.

        Parameters
        ----------
        mandatory_generation_of_conformers: bool
            If True, the conformers are generated and optimized before the descriptors are calculated.
        """
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcRadiusOfGyration

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Radius of Gyration calculation (length of 1).

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
          A numpy array of Molecule Representation of Structures based on Electron diffraction
        """
        fp = self.generate_descriptor(mol)
        return fp


class InertialShapeFactor(ThreeDimensionDescriptor):
    """
    Calculate Inertial Shape Factor
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        """
        Initialize the class.

        Parameters
        ----------
        mandatory_generation_of_conformers: bool
            If True, the conformers are generated and optimized before the descriptors are calculated.
        """
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcInertialShapeFactor

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Inertial Shape Factor (length of 1).

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
          A numpy array of the Inertial Shape Factor
        """
        fp = self.generate_descriptor(mol)
        return fp


class Eccentricity(ThreeDimensionDescriptor):
    """
    Calculate molecular eccentricity
    G. A. Arteca “Molecular Shape Descriptors” Reviews in Computational Chemistry vol 9
    https://doi.org/10.1002/9780470125861.ch5
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        """
        Initialize the class.

        Parameters
        ----------
        mandatory_generation_of_conformers: bool
            If True, the conformers are generated and optimized before the descriptors are calculated.
        """
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcEccentricity

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Eccentricity (length of 1).

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
          A numpy array of the Eccentricity
        """
        fp = self.generate_descriptor(mol)
        return fp


class Asphericity(ThreeDimensionDescriptor):
    """
    Calculate molecular Asphericity
    A. Baumgaertner, “Shapes of flexible vesicles” J. Chem. Phys. 98:7496 (1993)
    https://doi.org/10.1063/1.464689
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        """
        Initialize the class.

        Parameters
        ----------
        mandatory_generation_of_conformers: bool
            If True, the conformers are generated and optimized before the descriptors are calculated.
        """
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcAsphericity

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Asphericity (length of 1).

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
          A numpy array of the Asphericity
        """
        fp = self.generate_descriptor(mol)
        return fp


class SpherocityIndex(ThreeDimensionDescriptor):
    """
    Calculate molecular Spherocity Index
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        """
        Initialize the class.

        Parameters
        ----------
        mandatory_generation_of_conformers: bool
            If True, the conformers are generated and optimized before the descriptors are calculated.
        """
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcSpherocityIndex

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Spherocity Index (length of 1).

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
          A numpy array of the Spherocity Index
        """
        fp = self.generate_descriptor(mol)
        return fp


class PrincipalMomentsOfInertia(ThreeDimensionDescriptor):
    """
    Calculate Principal Moments of Inertia
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        """
        Initialize the class.

        Parameters
        ----------
        mandatory_generation_of_conformers: bool
            If True, the conformers are generated and optimized before the descriptors are calculated.
        """
        super().__init__(mandatory_generation_of_conformers)

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Principal Moments of Inertia (length of 3).

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        pmi: np.ndarray
          A numpy array of the Principal Moments of Inertia
        """

        try:
            has_conformers = check_atoms_coordinates(mol)
            if not has_conformers and self.mandatory_generation_of_conformers:
                mol = self.three_dimensional_generator.generate_conformers(mol)
                mol = self.three_dimensional_generator.optimize_molecular_geometry(mol)
            elif not has_conformers:
                raise PreConditionViolationException("molecule has no conformers")

            pmi1 = [rdMolDescriptors.CalcPMI1(mol)]
            pmi2 = [rdMolDescriptors.CalcPMI2(mol)]
            pmi3 = [rdMolDescriptors.CalcPMI3(mol)]

            pmi = pmi1 + pmi2 + pmi3

        except PreConditionViolationException as e:
            _no_conformers_message(e)
            raise e

        except Exception as e:
            print('error in smile: ' + str(mol))

            _no_conformers_message(e)
            pmi = np.empty(3, dtype=float)
            pmi[:] = np.NaN

        pmi = np.asarray(pmi, dtype=float)
        return pmi


class NormalizedPrincipalMomentsRatios(ThreeDimensionDescriptor):
    """
    Normalized principal moments ratios.
    Sauer and Schwarz JCIM 43:987-1003 (2003)
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        """
        Initialize the class.

        Parameters
        ----------
        mandatory_generation_of_conformers: bool
            If True, the conformers are generated and optimized before the descriptors are calculated.
        """
        super().__init__(mandatory_generation_of_conformers)

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Normalized Principal Moments Ratios (length of 2).

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        npr: np.ndarray
          A numpy array of the Normalized Principal Moments Ratios.
        """

        try:
            has_conformers = check_atoms_coordinates(mol)

            if not has_conformers and self.mandatory_generation_of_conformers:
                mol = self.three_dimensional_generator.generate_conformers(mol)
                mol = self.three_dimensional_generator.optimize_molecular_geometry(mol)

            elif not has_conformers:
                raise PreConditionViolationException("molecule has no conformers")

            npr1 = [rdMolDescriptors.CalcNPR1(mol)]
            npr2 = [rdMolDescriptors.CalcNPR2(mol)]

            npr = npr1 + npr2

        except PreConditionViolationException as e:
            _no_conformers_message(e)
            raise e

        except Exception as e:
            print('error in smile: ' + str(mol))
            _no_conformers_message(e)
            npr = np.empty(2, dtype=float)
            npr[:] = np.NaN

        npr = np.asarray(npr, dtype=float)
        return npr
