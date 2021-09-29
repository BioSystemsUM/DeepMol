import numpy as np
from mordred import Calculator, descriptors, get_descriptors_in_module, CPSA, GravitationalIndex, MoRSE, \
    GeometricalIndex, MomentOfInertia, PBF
from rdkit.Chem import Mol

from compoundFeaturization.baseFeaturizer import MolecularFeaturizer


class MordredFeaturizer(MolecularFeaturizer):
    """Mordred featurizers
    Includes all Mordred descriptors
    """

    def __init__(self, class_descriptors=descriptors, ignore_3D=True):

        super().__init__()
        self.descriptors = class_descriptors
        self.ignore_3D = ignore_3D

    def _featurize(self, mol: Mol) -> np.ndarray:
        """Mordred descriptors calculation (length of 1507)
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

            calculator = Calculator(self.descriptors, ignore_3D=self.ignore_3D)
            result = calculator(mol)
            result.fill_missing(value=np.NaN)

        except Exception as e:
            print('error in smile: ' + str(mol))
            result = np.empty(1826, dtype=float)
            result[:] = np.NaN

        result = np.asarray(result, dtype=np.float)

        return result


class Mordred3DFeaturizer(MolecularFeaturizer):
    """Mordred featurizers
    Includes all Mordred descriptors
    """

    def __init__(self):

        super().__init__()

    def _featurize(self, mol: Mol) -> np.ndarray:
        """Mordred 3D descriptors calculation (length of 80)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of 3D Mordred descriptors
        """

        try:
            cpsa = get_descriptors_in_module(CPSA)
            grav = get_descriptors_in_module(GravitationalIndex)
            morse = get_descriptors_in_module(MoRSE)
            geo_index = get_descriptors_in_module(GeometricalIndex)
            momi = get_descriptors_in_module(MomentOfInertia)
            pbf = get_descriptors_in_module(PBF)

            three_dimensional_descriptors = [cpsa, grav, morse, geo_index, momi, pbf]
            final_result = []

            for descript in three_dimensional_descriptors:
                calculator = Calculator(descript)
                result = calculator(mol)
                result.fill_missing(value=np.NaN)
                final_result.extend(result)


        except Exception as e:
            print('error in smile: ' + str(mol))
            final_result = np.empty(80, dtype=float)
            final_result[:] = np.NaN

        final_result = np.asarray(final_result, dtype=np.float)

        return final_result


class ChargedPartialSurfaceArea(MolecularFeaturizer):
    """Charged Partial Surface Area descriptor.
    Includes Charged Partial Surface Area Mordred descriptors
    """

    def __init__(self):

        super().__init__()

    def _featurize(self, mol: Mol) -> np.ndarray:
        """Charged Partial Surface Area calculation (length of 43)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of 3D Mordred descriptors
        """

        try:
            cpsa = get_descriptors_in_module(CPSA)

            calculator = Calculator(cpsa)
            result = calculator(mol)
            result.fill_missing(value=np.NaN)


        except Exception:
            print('error in smile: ' + str(mol))
            result = np.empty(43, dtype=float)
            result[:] = np.NaN

        result = np.asarray(result, dtype=np.float)

        return result


class MolecularGravitationalIndex(MolecularFeaturizer):
    """Molecular Gravitational Index featurizers
    """

    def __init__(self):

        super().__init__()

    def _featurize(self, mol: Mol) -> np.ndarray:
        """Molecular Gravitational Index calculation (length of 2)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of 3D Mordred descriptors
        """

        try:
            grav = get_descriptors_in_module(GravitationalIndex)

            calculator = Calculator(grav)
            result = calculator(mol)
            result.fill_missing(value=np.NaN)


        except Exception:
            print('error in smile: ' + str(mol))
            result = np.empty(2, dtype=float)
            result[:] = np.NaN

        result = np.asarray(result, dtype=np.float)

        return result


class MORSE(MolecularFeaturizer):
    """MoRSE descriptor.
    """

    def __init__(self):

        super().__init__()

    def _featurize(self, mol: Mol) -> np.ndarray:
        """Charged Partial Surface Area calculation (length of 5)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of 3D Mordred descriptors
        """

        try:
            morse = get_descriptors_in_module(MoRSE)

            calculator = Calculator(morse)
            result = calculator(mol)
            result.fill_missing(value=np.NaN)


        except Exception as e:
            print('error in smile: ' + str(mol))
            result = np.empty(5, dtype=float)
            result[:] = np.NaN

        result = np.asarray(result, dtype=np.float)

        return result


class MolecularGeometricalIndex(MolecularFeaturizer):
    """Molecular Geometrical Index descriptor.
    Includes Molecular Geometrical Index Mordred descriptors
    """

    def __init__(self):

        super().__init__()

    def _featurize(self, mol: Mol) -> np.ndarray:
        """Molecular Geometrical Index calculation (length of 4)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of Molecular Geometrical Indexes
        """

        try:
            geo_index = get_descriptors_in_module(GeometricalIndex)

            calculator = Calculator(geo_index)
            result = calculator(mol)
            result.fill_missing(value=np.NaN)


        except Exception:
            print('error in smile: ' + str(mol))
            result = np.empty(4, dtype=float)
            result[:] = np.NaN

        result = np.asarray(result, dtype=np.float)

        return result


class MolecularMomentOfInertia(MolecularFeaturizer):
    """
    Molecular Moment Of Inertia
    Includes Molecular Moment Of Inertia from Mordred package
    """

    def __init__(self):

        super().__init__()

    def _featurize(self, mol: Mol) -> np.ndarray:
        """Molecular Moment Of Inertia (length of 3)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of Molecular Moment Of Inertia
        """

        try:
            momi = get_descriptors_in_module(MomentOfInertia)

            calculator = Calculator(momi)
            result = calculator(mol)
            result.fill_missing(value=np.NaN)


        except Exception:
            print('error in smile: ' + str(mol))
            result = np.empty(3, dtype=float)
            result[:] = np.NaN

        result = np.asarray(result, dtype=np.float)

        return result


class PlaneOfBestFit(MolecularFeaturizer):
    """
    Plane Of Best Fit
    Includes Plane Of Best Fit
    """

    def __init__(self):

        super().__init__()

    def _featurize(self, mol: Mol) -> np.ndarray:
        """Plane Of Best Fit calculation (length of 3)
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of Molecular Moment Of Inertia descriptors
        """

        try:
            pbf = get_descriptors_in_module(PBF)

            calculator = Calculator(pbf)
            result = calculator(mol)
            result.fill_missing(value=np.NaN)


        except Exception:
            print('error in smile: ' + str(mol))
            result = np.empty(3, dtype=float)
            result[:] = np.NaN

        result = np.asarray(result, dtype=np.float)

        return result
