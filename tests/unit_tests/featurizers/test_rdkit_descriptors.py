import os
from copy import copy
from unittest import TestCase

from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolAlign import AlignMol

from deepmol.compound_featurization.rdkit_descriptors import ThreeDimensionalMoleculeGenerator, All3DDescriptors, \
    AutoCorr3D, \
    RadialDistributionFunction, PlaneOfBestFit, MORSE, WHIM, RadiusOfGyration, InertialShapeFactor, Eccentricity, \
    Asphericity, SpherocityIndex, PrincipalMomentsOfInertia, NormalizedPrincipalMomentsRatios, \
    generate_conformers_to_sdf_file, TwoDimensionDescriptors, get_all_3D_descriptors
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase

import numpy as np


class Test2DDescriptors(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        TwoDimensionDescriptors().featurize(self.mini_dataset_to_test)
        self.assertEqual(5, self.mini_dataset_to_test.X.shape[0])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mini_dataset_to_test.mols)
        to_add = np.zeros(4)
        ids_to_add = np.array([i for i in range(dataset_rows_number, dataset_rows_number + 4)])

        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))
        self.mini_dataset_to_test.ids = np.concatenate((self.mini_dataset_to_test.ids, ids_to_add))

        dataset = copy(self.mini_dataset_to_test)
        TwoDimensionDescriptors().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

    def test_with_invalid_smiles(self):
        TwoDimensionDescriptors().featurize(self.dataset_invalid_smiles)


class Test3DDescriptors(FeaturizerTestCase, TestCase):

    def molecular_geometry_generation_and_optimization(self, smiles, method, etkg_version, generator):
        mol_raw = MolFromSmiles(smiles)
        mol_raw2 = MolFromSmiles(smiles)
        new_mol = MolFromSmiles(smiles)
        new_mol = generator.generate_conformers(new_mol, etkg_version)
        conformer_1_before = new_mol.GetConformer(1)
        conformers = new_mol.GetConformers()
        self.assertEqual(len(conformers), 10)
        new_mol = generator.optimize_molecular_geometry(new_mol, method)

        conformer_1_after = new_mol.GetConformer(1)

        mol_raw = Chem.AddHs(mol_raw)
        mol_raw2 = Chem.AddHs(mol_raw2)

        mol_raw.AddConformer(conformer_1_before)
        mol_raw2.AddConformer(conformer_1_after)

        rmsd = AlignMol(mol_raw, mol_raw2)
        self.assertAlmostEqual(rmsd, 0, delta=10)

    def test_before_featurization(self):
        generator = ThreeDimensionalMoleculeGenerator()

        mol = MolFromSmiles("CC(CC(C)(O)C=C)=CC=CC")
        mol = generator.generate_conformers(mol, 1)
        conformers = mol.GetConformers()
        self.assertEqual(len(conformers), 5)

        new_generator = ThreeDimensionalMoleculeGenerator(n_conformations=10)
        mol_raw = MolFromSmiles("CC(CC(C)(O)C=C)=CC=C")

        conformers = mol_raw.GetConformers()
        self.assertEqual(len(conformers), 0)

        ETKG_version_combinations = [1, 2, 3]
        geometry_optimization_method = ["MMFF94", "UFF"]
        for smiles in self.mini_dataset_to_test.mols:
            for version in ETKG_version_combinations:
                for method in geometry_optimization_method:
                    self.molecular_geometry_generation_and_optimization(smiles, method, version, new_generator)

    def test_featurize(self):
        dataset_rows_number = len(self.mini_dataset_to_test.mols)
        All3DDescriptors(mandatory_generation_of_conformers=True).featurize(self.mini_dataset_to_test)
        self.assertEqual(dataset_rows_number, self.mini_dataset_to_test.X.shape[0])
        # assert that all 3d descriptors are always computed in the same order
        mol = ThreeDimensionalMoleculeGenerator().generate_conformers(self.mini_dataset_to_test.mols[0])
        t1 = get_all_3D_descriptors(mol)
        t2 = get_all_3D_descriptors(mol)
        t3 = get_all_3D_descriptors(mol)
        for i in range(len(t1)):
            self.assertEqual(t1[i], t2[i])
            self.assertEqual(t1[i], t3[i])


    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mini_dataset_to_test.mols)
        to_add = np.zeros(4)
        ids_to_add = np.array([i for i in range(dataset_rows_number, dataset_rows_number + 4)])

        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))
        self.mini_dataset_to_test.ids = np.concatenate((self.mini_dataset_to_test.ids, ids_to_add))

        dataset = copy(self.mini_dataset_to_test)
        All3DDescriptors(mandatory_generation_of_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        AutoCorr3D(mandatory_generation_of_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        RadialDistributionFunction(mandatory_generation_of_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        PlaneOfBestFit(mandatory_generation_of_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        MORSE(mandatory_generation_of_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        WHIM(mandatory_generation_of_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        RadiusOfGyration(mandatory_generation_of_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        InertialShapeFactor(mandatory_generation_of_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        Eccentricity(mandatory_generation_of_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        Asphericity(mandatory_generation_of_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        SpherocityIndex(mandatory_generation_of_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        PrincipalMomentsOfInertia(mandatory_generation_of_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        NormalizedPrincipalMomentsRatios(mandatory_generation_of_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

    def test_featurize_to_fail(self):

        with self.assertRaises(SystemExit) as cm:
            All3DDescriptors(mandatory_generation_of_conformers=False).featurize(self.mini_dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            AutoCorr3D(mandatory_generation_of_conformers=False).featurize(self.mini_dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            RadialDistributionFunction(mandatory_generation_of_conformers=False).featurize(self.mini_dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            PlaneOfBestFit(mandatory_generation_of_conformers=False).featurize(self.mini_dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            MORSE(mandatory_generation_of_conformers=False).featurize(self.mini_dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            WHIM(mandatory_generation_of_conformers=False).featurize(self.mini_dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            RadiusOfGyration(mandatory_generation_of_conformers=False).featurize(self.mini_dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            InertialShapeFactor(mandatory_generation_of_conformers=False).featurize(self.mini_dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            Eccentricity(mandatory_generation_of_conformers=False).featurize(self.mini_dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            Asphericity(mandatory_generation_of_conformers=False).featurize(self.mini_dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            SpherocityIndex(mandatory_generation_of_conformers=False).featurize(self.mini_dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            PrincipalMomentsOfInertia(mandatory_generation_of_conformers=False).featurize(self.mini_dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            NormalizedPrincipalMomentsRatios(mandatory_generation_of_conformers=False).featurize(
                self.mini_dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

    def test_generate_conformers_and_export(self):

        generate_conformers_to_sdf_file(self.mini_dataset_to_test, "temp.sdf", n_conformations=1, max_iterations=1)
        os.remove("temp.sdf")
