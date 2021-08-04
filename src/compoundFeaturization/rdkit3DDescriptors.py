import inspect
import queue
from threading import Thread
from typing import Union

from rdkit import Chem
from rdkit.Chem import Mol, rdMolDescriptors, AllChem, MolFromSmiles
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMoleculeConfs
from rdkit.Chem.rdmolops import RemoveHs

from Datasets.Datasets import Dataset
from compoundFeaturization.baseFeaturizer import MolecularFeaturizer
import numpy as np

import sys

import traceback


def _no_conformers_message(e):
    exc = traceback.format_exc()

    if isinstance(e, RuntimeError) and "molecule has no conformers" in exc \
          or isinstance(e, ValueError) and "Bad Conformer Id" in exc:
        print("You have to generate molecular conformers for each molecule. \n"
                           "You can execute the following method: \n"
                           "rdkit3DDescriptors.generate_conformers_to_sdf_file(dataset: Dataset, file_path: str,"
                           " n_conformations: int,max_iterations: int, threads: int, timeout_per_molecule: int) \n"
                           "The result will be stored in a SDF format file which can be loaded with the "
                           "method: loaders.Loaders.SDFLoader()")

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
    conf = mol.GetConformer()
    position = []
    for i in range(conf.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        position.append([pos.x, pos.y, pos.z])
    position = np.array(position)
    if not np.any(position):
        raise RuntimeError("molecule has no conformers")


def get_all_3D_descriptors(mol):
    """
    Method that lists all the methods and uses them to featurize the whole set.
    """

    size = 639

    current_module = sys.modules[__name__]

    all_descriptors = np.empty(0, dtype=float)
    for name, descriptor_function in inspect.getmembers(current_module, inspect.isclass):
        try:
            parents = [parent_function.__name__ for parent_function in inspect.getmro(descriptor_function)]

            if name not in ["All3DDescriptors", "MolecularFeaturizer", "GETAWAY"] and \
                    "MolecularFeaturizer" in parents:

                descriptor_values = descriptor_function()._featurize(mol)

                if not np.any(np.isnan(descriptor_values)):
                    all_descriptors = np.concatenate((all_descriptors, descriptor_values))

                else:
                    print('error in molecule: ' + str(mol))
                    all_descriptors = np.empty(size, dtype=float)
                    all_descriptors[:] = np.NaN
                    break

        except Exception:
            print('error in molecule: ' + str(mol))
            all_descriptors = np.empty(size, dtype=float)
            all_descriptors[:] = np.NaN
            break

    return all_descriptors


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

        new_mol = RemoveHs(new_mol)

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

        if "MMFF" in mode:
            AllChem.MMFFOptimizeMoleculeConfs(mol,
                                              maxIters=self.max_iterations,
                                              numThreads=self.threads, mmffVariant=mode)

        elif mode == "UFF":

            UFFOptimizeMoleculeConfs(mol, maxIters=self.max_iterations,
                                     numThreads=self.threads)

        return mol


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

    generator = ThreeDimensionalMoleculeGenerator(max_iterations, n_conformations, threads, timeout_per_molecule)

    my_queue = queue.Queue()

    def store_in_queue(f):
        def wrapper(*args):
            my_queue.put(f(*args))

        return wrapper

    @store_in_queue
    def generate_conformers(new_mol: Union[Mol, str], ETKG_version: int = 1, optimization_mode: str = "MMFF94"):

        if isinstance(new_mol, str):
            new_mol = MolFromSmiles(new_mol)

        new_mol = Chem.AddHs(new_mol)

        new_mol = generator.generate_conformers(new_mol, ETKG_version)
        new_mol = generator.optimize_molecular_geometry(new_mol, optimization_mode)

        return new_mol

    mol_set = dataset.mols

    final_set_with_conformations = []

    for i in range(mol_set.shape[0]):
        action_thread = Thread(target=generate_conformers, args=(mol_set[i], ETKG_version, optimization_mode,))
        action_thread.start()

        m2 = my_queue.get(True, timeout=timeout_per_molecule)

        label = dataset.y[i]
        mol_id = dataset.ids[i]
        m2.SetProp("_Class", "%f" % label)
        m2.SetProp("_ID", "%f" % mol_id)

        final_set_with_conformations.append(m2)

    writer = Chem.SDWriter(file_path)

    for mol in final_set_with_conformations:
        writer.write(mol)

    writer.close()


class All3DDescriptors(MolecularFeaturizer):

    def __init__(self):
        super().__init__()

    def _featurize(self, mol: Mol) -> np.ndarray:
        """ Featurization of a molecule with all rdkit 3D featurizers
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of all 3D featurizers from rdkit
        """

        size = 639

        try:
            check_atoms_coordinates(mol)
            fp = get_all_3D_descriptors(mol)

        except Exception as e:

            print('error in smile: ' + str(mol))

            _no_conformers_message(e)
            fp = np.empty(size, dtype=float)
            fp[:] = np.NaN

        fp = np.asarray(fp, dtype=np.float)

        return fp


class AutoCorr3D(MolecularFeaturizer):
    """AutoCorr3D.
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self):

        super().__init__()

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

        try:
            check_atoms_coordinates(mol)
            fp = rdMolDescriptors.CalcAUTOCORR3D(mol)

        except Exception as e:

            print('error in smile: ' + str(mol))
            _no_conformers_message(e)

            fp = np.empty(80, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp


class RadialDistributionFunction(MolecularFeaturizer):
    """Radial distribution function
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self):

        super().__init__()

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

        try:
            check_atoms_coordinates(mol)
            fp = rdMolDescriptors.CalcRDF(mol)

        except Exception as e:
            print('error in smile: ' + str(mol))
            _no_conformers_message(e)

            fp = np.empty(210, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp


class PlaneOfBestFit(MolecularFeaturizer):
    """Plane of best fit
    Nicholas C. Firth, Nathan Brown, and Julian Blagg, JCIM 52:2516-25
    """

    def __init__(self):

        super().__init__()

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

        try:
            check_atoms_coordinates(mol)
            fp = [rdMolDescriptors.CalcPBF(mol)]

        except Exception as e:
            print('error in smile: ' + str(mol))
            _no_conformers_message(e)
            fp = np.empty(1, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp


class MORSE(MolecularFeaturizer):
    """Molecule Representation of Structures based on Electron diffraction descriptors
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self):

        super().__init__()

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

        try:
            check_atoms_coordinates(mol)
            fp = rdMolDescriptors.CalcMORSE(mol)

        except Exception as e:
            print('error in smile: ' + str(mol))
            _no_conformers_message(e)
            fp = np.empty(224, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp


class WHIM(MolecularFeaturizer):
    """
    WHIM descriptors vector
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self):

        super().__init__()

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

        try:
            check_atoms_coordinates(mol)
            fp = rdMolDescriptors.CalcWHIM(mol)

        except Exception as e:
            print('error in smile: ' + str(mol))

            _no_conformers_message(e)
            fp = np.empty(114, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp


class RadiusOfGyration(MolecularFeaturizer):
    """
    Calculate Radius of Gyration
    G. A. Arteca “Molecular Shape Descriptors” Reviews in Computational Chemistry vol 9
    https://doi.org/10.1002/9780470125861.ch5
    """

    def __init__(self):

        super().__init__()

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

        try:
            check_atoms_coordinates(mol)
            fp = [rdMolDescriptors.CalcRadiusOfGyration(mol)]

        except Exception as e:
            print('error in smile: ' + str(mol))

            _no_conformers_message(e)
            fp = np.empty(1, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp


class InertialShapeFactor(MolecularFeaturizer):
    """
    Calculate Inertial Shape Factor
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self):

        super().__init__()

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

        try:
            check_atoms_coordinates(mol)
            fp = [rdMolDescriptors.CalcInertialShapeFactor(mol)]

        except Exception as e:
            print('error in smile: ' + str(mol))
            _no_conformers_message(e)
            fp = np.empty(1, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp


class Eccentricity(MolecularFeaturizer):
    """
    Calculate molecular eccentricity
    G. A. Arteca “Molecular Shape Descriptors” Reviews in Computational Chemistry vol 9
    https://doi.org/10.1002/9780470125861.ch5
    """

    def __init__(self):

        super().__init__()

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

        try:
            check_atoms_coordinates(mol)
            fp = [rdMolDescriptors.CalcEccentricity(mol)]

        except Exception as e:
            print('error in smile: ' + str(mol))
            _no_conformers_message(e)
            fp = np.empty(1, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp


class Asphericity(MolecularFeaturizer):
    """
    Calculate molecular Asphericity
    A. Baumgaertner, “Shapes of flexible vesicles” J. Chem. Phys. 98:7496 (1993)
    https://doi.org/10.1063/1.464689
    """

    def __init__(self):

        super().__init__()

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

        try:
            check_atoms_coordinates(mol)
            fp = [rdMolDescriptors.CalcAsphericity(mol)]

        except Exception as e:
            print('error in smile: ' + str(mol))
            _no_conformers_message(e)
            fp = np.empty(1, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp


class SpherocityIndex(MolecularFeaturizer):
    """
    Calculate molecular Spherocity Index
    Todeschini and Consoni “Descriptors from Molecular Geometry” Handbook of Chemoinformatics
    https://doi.org/10.1002/9783527618279.ch37
    """

    def __init__(self):

        super().__init__()

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

        try:
            check_atoms_coordinates(mol)
            fp = [rdMolDescriptors.CalcSpherocityIndex(mol)]

        except Exception as e:
            print('error in smile: ' + str(mol))
            _no_conformers_message(e)
            fp = np.empty(1, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp


class PrincipalMomentsOfInertia(MolecularFeaturizer):
    """
    Calculate Principal Moments of Inertia

    """

    def __init__(self):

        super().__init__()

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
            check_atoms_coordinates(mol)

            pmi1 = [rdMolDescriptors.CalcPMI1(mol)]
            pmi2 = [rdMolDescriptors.CalcPMI2(mol)]
            pmi3 = [rdMolDescriptors.CalcPMI3(mol)]

            pmi = pmi1 + pmi2 + pmi3

        except Exception as e:
            print('error in smile: ' + str(mol))

            _no_conformers_message(e)
            pmi = np.empty(3, dtype=float)
            pmi[:] = np.NaN

        pmi = np.asarray(pmi, dtype=np.float)

        return pmi


class NormalizedPrincipalMomentsRatios(MolecularFeaturizer):
    """
    Normalized principal moments ratios
    Sauer and Schwarz JCIM 43:987-1003 (2003)

    """

    def __init__(self):

        super().__init__()

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
            check_atoms_coordinates(mol)
            npr1 = [rdMolDescriptors.CalcNPR1(mol)]
            npr2 = [rdMolDescriptors.CalcNPR2(mol)]

            npr = npr1 + npr2

        except Exception as e:
            print('error in smile: ' + str(mol))

            _no_conformers_message(e)
            npr = np.empty(2, dtype=float)
            npr[:] = np.NaN

        npr = np.asarray(npr, dtype=np.float)

        return npr
