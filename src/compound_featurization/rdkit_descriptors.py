import inspect
from typing import Union

from numpy import float64, int64
from rdkit import Chem
from rdkit.Chem import Mol, rdMolDescriptors, AllChem, MolFromSmiles, Descriptors
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMoleculeConfs
from rdkit.ML.Descriptors import MoleculeDescriptors

from datasets.datasets import Dataset
from compound_featurization.base_featurizer import MolecularFeaturizer
import numpy as np

import sys

import traceback

from utils.errors import PreConditionViolationException


def _no_conformers_message(e):
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
        Returns True if molecules is OK and False if molecule contains zero coordinates.
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

    def __init__(self, n_conformations: int = 20,
                 max_iterations: int = 2000,
                 threads: int = 1,
                 timeout_per_molecule: int = 40):
        """
        Class to generate three dimensional conformers and optimize them

        Parameters
        ----------
        n_conformations: int
          Number of conformations to be generated per molecule
        max_iterations: int
          maximum of iterations when optimizing molecular geometry
        threads: int
          number of threads to be used
        timeout_per_molecule: int
          time to be spent in each molecule

        """

        self.max_iterations = max_iterations
        self.n_conformations = n_conformations
        self.threads = threads
        self.timeout_per_molecule = timeout_per_molecule

    @staticmethod
    def check_if_mol_has_explicit_hydrogens(new_mol: Mol):

        atoms = new_mol.GetAtoms()
        for atom in atoms:
            number_hydrogens = atom.GetNumExplicitHs()
            if number_hydrogens != 0:
                return True
        return False

    def generate_conformers(self, new_mol: Mol, ETKDG_version: int = 1, **kwargs):
        """
        method to generate three dimensional conformers

        Parameters
        ----------
        new_mol: Mol
          Mol object from rdkit
        ETKDG_version: int
          version of the experimental-torsion-knowledge distance geometry (ETKDG) algorithm

        """

        new_mol = Chem.AddHs(new_mol)

        if ETKDG_version == 1:
            AllChem.EmbedMultipleConfs(new_mol, numConfs=self.n_conformations,
                                       params=AllChem.ETKDG(), **kwargs)

        elif ETKDG_version == 2:
            AllChem.EmbedMultipleConfs(new_mol, numConfs=self.n_conformations,
                                       params=Chem.rdDistGeom.ETKDGv2(), **kwargs)

        elif ETKDG_version == 3:
            AllChem.EmbedMultipleConfs(new_mol, numConfs=self.n_conformations,
                                       params=Chem.rdDistGeom.ETKDGv3(), **kwargs)

        else:
            print("Choose ETKDG's valid version (1,2 or 3)")
            return None

        return new_mol

    def optimize_molecular_geometry(self, mol: Mol, mode: str = "MMFF94"):

        """
        Class to generate three dimensional conformers

        Parameters
        ----------
        mol: Mol
          Mol object from rdkit
        mode: int
          mode for the molecular geometry optimization (MMFF or UFF)

        """

        mol = Chem.AddHs(mol)

        if "MMFF" in mode:
            AllChem.MMFFOptimizeMoleculeConfs(mol,
                                              maxIters=self.max_iterations,
                                              numThreads=self.threads, mmffVariant=mode)

        elif mode == "UFF":

            UFFOptimizeMoleculeConfs(mol, maxIters=self.max_iterations,
                                     numThreads=self.threads)

        return mol


def get_all_3D_descriptors(mol):
    """
    Method that lists all the methods and uses them to featurize the whole set.
    """

    size = 639

    current_module = sys.modules[__name__]

    all_descriptors = np.empty(0, dtype=float)
    for name, featurizer_function in inspect.getmembers(current_module, inspect.isclass):
        try:
            if issubclass(featurizer_function, ThreeDimensionDescriptor) and \
                    issubclass(featurizer_function, MolecularFeaturizer) and \
                    not name in [All3DDescriptors.__name__, ThreeDimensionDescriptor.__name__]:

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
                        new_mol: Union[Mol, str], ETKG_version: int = 1,
                        optimization_mode: str = "MMFF94"):
    if isinstance(new_mol, str):
        new_mol = MolFromSmiles(new_mol)

    new_mol = generator.generate_conformers(new_mol, ETKG_version)
    new_mol = generator.optimize_molecular_geometry(new_mol, optimization_mode)
    return new_mol


# TODO : check whether sdf file is being correctly exported for multi-class classification
def generate_conformers_to_sdf_file(dataset: Dataset, file_path: str, n_conformations: int = 20,
                                    max_iterations: int = 2000, threads: int = 1, timeout_per_molecule: int = 12,
                                    ETKG_version: int = 1, optimization_mode: str = "MMFF94"):
    """
    Generate conformers using the experimental-torsion-knowledge distance geometry (ETKDG) algorithm from RDKit,
    optimize them and save in a SDF file

        Parameters
        ----------
        dataset: Dataset
          DeepMol Dataset object
        file_path: str
          file_path where the conformers will be saved
        n_conformations: int
          the number of conformations per molecule
        max_iterations: int
          maximum number of iterations for the molecule's conformers optimization
        threads: int
          number of threads
        timeout_per_molecule: int
          the number of seconds in which the conformers are to be generated
        ETKG_version: int
          version of the experimental-torsion-knowledge distance geometry (ETKDG) algorithm
        optimization_mode: str
          mode for the molecular geometry optimization (MMFF or UFF)
    """

    def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='|'):
        """
      Call in a loop to create terminal progress bar
      @params:
          iteration   - Required  : current iteration (Int)
          total       - Required  : total iterations (Int)
          prefix      - Optional  : prefix string (Str)
          suffix      - Optional  : suffix string (Str)
          decimals    - Optional  : positive number of decimals in percent complete (Int)
          length      - Optional  : character length of bar (Int)
          fill        - Optional  : bar fill character (Str)
          printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
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
            m2 = generate_conformers(generator, mol_set[i], ETKG_version, optimization_mode)

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

    def __init__(self):
        super().__init__()

    def _featurize(self, mol: Mol):

        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        #header = calc.GetDescriptorNames()

        try:
            descriptors = calc.CalcDescriptors(mol)
            if np.isnan(np.sum(descriptors)):
                raise Exception

        except Exception as e:

            print('error in smile: ' + str(mol))
            _no_conformers_message(e)

            descriptors = np.empty(208, dtype=float64)
            descriptors[:] = np.NaN

        descriptors = np.array(descriptors, dtype=np.float64)

        return descriptors


class ThreeDimensionDescriptor(MolecularFeaturizer):

    def __init__(self, mandatory_generation_of_conformers):
        self.descriptor_function = None
        self.mandatory_generation_of_conformers = mandatory_generation_of_conformers
        if self.mandatory_generation_of_conformers:
            self.three_dimensional_generator = ThreeDimensionalMoleculeGenerator()
        super().__init__()

    @property
    def descriptor_function(self):
        return self._descriptor_function

    @descriptor_function.setter
    def descriptor_function(self, function: callable):
        self._descriptor_function = function

    def generate_descriptor(self, mol):
        try:
            has_conformers = check_atoms_coordinates(mol)

            if not has_conformers and self.mandatory_generation_of_conformers:
                mol = self.three_dimensional_generator.generate_conformers(mol)
                mol = self.three_dimensional_generator.optimize_molecular_geometry(mol)

            elif not has_conformers:
                raise PreConditionViolationException("molecule has no conformers")

            fp = self.descriptor_function(mol)
            if any([isinstance(fp, fp_type) for fp_type in [str, int, float64, float, int64]]):
                fp = [fp]

        except PreConditionViolationException as e:

            _no_conformers_message(e)
            raise e

        except Exception as e:

            print('error in smile: ' + str(mol))
            _no_conformers_message(e)

            fp = np.empty(80, dtype=float)
            fp[:] = np.NaN

        fp = np.asarray(fp, dtype=np.float)

        return fp

    def _featurize(self, mol: Mol):
        raise NotImplementedError


class All3DDescriptors(MolecularFeaturizer):

    def __init__(self, mandatory_generation_of_conformers=False):
        self.generate_conformers = mandatory_generation_of_conformers
        if self.generate_conformers:
            self.three_dimensional_generator = ThreeDimensionalMoleculeGenerator()

        super().__init__()

    def _featurize(self, mol: Mol) -> np.ndarray:
        """ Featurization of a molecule with all rdkit 3D descriptors
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of all 3D descriptors from rdkit
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

        fp = np.asarray(fp, dtype=np.float)

        return fp


class AutoCorr3D(ThreeDimensionDescriptor):
    """AutoCorr3D.
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcAUTOCORR3D

    def _featurize(self, mol: Mol) -> np.ndarray:
        """3D Autocorrelation descriptors vector calculation (length of 80)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of 3D Autocorrelation descriptors
        """

        fp = self.generate_descriptor(mol)
        return fp


class RadialDistributionFunction(ThreeDimensionDescriptor):
    """Radial distribution function
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcRDF

    def _featurize(self, mol: Mol) -> np.ndarray:
        """ Radial distribution function descriptors calculation (length of 210)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of Radial distribution function results.
        """

        fp = self.generate_descriptor(mol)

        return fp


class PlaneOfBestFit(ThreeDimensionDescriptor):
    """Plane of best fit
    Nicholas C. Firth, Nathan Brown, and Julian Blagg, JCIM 52:2516-25
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcPBF

    def _featurize(self, mol: Mol) -> np.ndarray:
        """ Radial distribution function descriptors calculation
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array with the Plane of best fit.
        """

        fp = self.generate_descriptor(mol)

        return fp


class MORSE(ThreeDimensionDescriptor):
    """Molecule Representation of Structures based on Electron diffraction descriptors
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self, mandatory_generation_of_conformers=False):
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcMORSE

    def _featurize(self, mol: Mol) -> np.ndarray:
        """ Molecule Representation of Structures based on Electron diffraction descriptors calculation (length of 224)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
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
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcWHIM

    def _featurize(self, mol: Mol) -> np.ndarray:
        """ WHIM descriptors calculation (length of 114)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
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
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcRadiusOfGyration

    def _featurize(self, mol: Mol) -> np.ndarray:
        """ Radius of Gyration calculation (length of 1)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
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
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcInertialShapeFactor

    def _featurize(self, mol: Mol) -> np.ndarray:
        """ Inertial Shape Factor (length of 1)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
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
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcEccentricity

    def _featurize(self, mol: Mol) -> np.ndarray:
        """ Eccentricity (length of 1)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
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
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcEccentricity

    def _featurize(self, mol: Mol) -> np.ndarray:
        """ Asphericity (length of 1)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
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
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcEccentricity

    def _featurize(self, mol: Mol) -> np.ndarray:
        """ Spherocity Index (length of 1)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of the Spherocity Index
        """

        fp = self.generate_descriptor(mol)

        return fp


class PrincipalMomentsOfInertia(ThreeDimensionDescriptor):
    """
    Calculate Principal Moments of Inertia

    """

    def __init__(self, mandatory_generation_of_conformers=False):
        super().__init__(mandatory_generation_of_conformers)
        self.descriptor_function = rdMolDescriptors.CalcEccentricity

    def _featurize(self, mol: Mol) -> np.ndarray:
        """ Principal Moments of Inertia (length of 3)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
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

        pmi = np.asarray(pmi, dtype=np.float)

        return pmi


class NormalizedPrincipalMomentsRatios(ThreeDimensionDescriptor):
    """
    Normalized principal moments ratios
    Sauer and Schwarz JCIM 43:987-1003 (2003)

    """

    def __init__(self, mandatory_generation_of_conformers=False):

        super().__init__(mandatory_generation_of_conformers)

    def _featurize(self, mol: Mol) -> np.ndarray:
        """ Normalized Principal Moments Ratios (length of 2)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of the Normalized Principal Moments Ratios
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

        npr = np.asarray(npr, dtype=np.float)

        return npr
