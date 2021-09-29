import os
from copy import copy
from unittest import TestCase

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolAlign import AlignMol

from compoundFeaturization.rdkit3DDescriptors import ThreeDimensionalMoleculeGenerator, All3DDescriptors, AutoCorr3D, \
    RadialDistributionFunction, PlaneOfBestFit, MORSE, WHIM, RadiusOfGyration, InertialShapeFactor, Eccentricity, \
    Asphericity, SpherocityIndex, PrincipalMomentsOfInertia, NormalizedPrincipalMomentsRatios, \
    generate_conformers_to_sdf_file
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase

import numpy as np


class Test3DDescriptors(FeaturizerTestCase, TestCase):

    def molecular_geometry_generation_and_optimization(self, smiles, method, ETKG_version, generator):
        mol_raw = MolFromSmiles(smiles)
        mol_raw2 = MolFromSmiles(smiles)
        new_mol = MolFromSmiles(smiles)
        new_mol = generator.generate_conformers(new_mol, ETKG_version)
        conformer_1_before = new_mol.GetConformer(1)
        self.assertEquals(len(new_mol.GetConformers()), 10)
        new_mol = generator.optimize_molecular_geometry(new_mol, method)

        conformer_1_after = new_mol.GetConformer(1)

        mol_raw.AddConformer(conformer_1_before)
        mol_raw2.AddConformer(conformer_1_after)

        rmsd = AlignMol(mol_raw, mol_raw2)
        self.assertAlmostEqual(rmsd, 0, delta=0.01)

    def test_before_featurization(self):
        generator = ThreeDimensionalMoleculeGenerator()

        mol = MolFromSmiles("CC(CC(C)(O)C=C)=CC=CC")
        mol = generator.generate_conformers(mol, 1)
        self.assertEquals(len(mol.GetConformers()), 20)

        new_generator = ThreeDimensionalMoleculeGenerator(n_conformations=10)
        mol_raw = MolFromSmiles("CC(CC(C)(O)C=C)=CC=C")
        self.assertEquals(mol_raw.GetConformers(), ())

        ETKG_version_combinations = [1, 2, 3]
        geometry_optimization_method = ["MMFF94", "UFF"]
        for smiles in self.dataset_to_test.mols:
            for version in ETKG_version_combinations:
                for method in geometry_optimization_method:
                    self.molecular_geometry_generation_and_optimization(smiles, method, version, new_generator)

    def test_featurize(self):
        dataset_rows_number = len(self.dataset_to_test.mols)
        All3DDescriptors(generate_conformers=True).featurize(self.dataset_to_test)
        self.assertEqual(dataset_rows_number, self.dataset_to_test.X.shape[0])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.dataset_to_test.mols)
        to_add = np.zeros(4)

        self.dataset_to_test.mols = np.concatenate((self.dataset_to_test.mols, to_add))
        self.dataset_to_test.y = np.concatenate((self.dataset_to_test.y, to_add))

        dataset = copy(self.dataset_to_test)
        All3DDescriptors(generate_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        AutoCorr3D(generate_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        RadialDistributionFunction(generate_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        PlaneOfBestFit(generate_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        MORSE(generate_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        WHIM(generate_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        RadiusOfGyration(generate_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        InertialShapeFactor(generate_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        Eccentricity(generate_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        Asphericity(generate_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        SpherocityIndex(generate_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        PrincipalMomentsOfInertia(generate_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        NormalizedPrincipalMomentsRatios(generate_conformers=True).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

    def test_featurize_to_fail(self):

        with self.assertRaises(SystemExit) as cm:
            All3DDescriptors().featurize(self.dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            AutoCorr3D().featurize(self.dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            RadialDistributionFunction().featurize(self.dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            PlaneOfBestFit().featurize(self.dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            MORSE().featurize(self.dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            WHIM().featurize(self.dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            RadiusOfGyration().featurize(self.dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            InertialShapeFactor().featurize(self.dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            Eccentricity().featurize(self.dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            Asphericity().featurize(self.dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            SpherocityIndex().featurize(self.dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            PrincipalMomentsOfInertia().featurize(self.dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

        with self.assertRaises(SystemExit) as cm:
            NormalizedPrincipalMomentsRatios().featurize(self.dataset_to_test)

        self.assertEqual(cm.exception.code, 1)

    def test_generate_conformers_and_export(self):

        generate_conformers_to_sdf_file(self.dataset_to_test, "temp.sdf")
        os.remove("temp.sdf")