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


class Test2DDescriptors(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        TwoDimensionDescriptors().featurize(self.mock_dataset, inplace=True)
        self.assertEqual(7, self.mock_dataset._X.shape[0])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mock_dataset_with_invalid.mols) - 1  # one mol has invalid smiles

        dataset = copy(self.mock_dataset_with_invalid)
        TwoDimensionDescriptors().featurize(dataset, inplace=True)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])


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
        for smiles in self.mock_dataset.smiles:
            for version in ETKG_version_combinations:
                for method in geometry_optimization_method:
                    self.molecular_geometry_generation_and_optimization(smiles, method, version, new_generator)

    def test_featurize(self):
        dataset_rows_number = len(self.mock_dataset.mols)
        All3DDescriptors(mandatory_generation_of_conformers=True).featurize(self.mock_dataset, inplace=True)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])
        # assert that all 3d descriptors are always computed in the same order
        mol = ThreeDimensionalMoleculeGenerator().generate_conformers(self.mock_dataset.mols[0])
        t1 = get_all_3D_descriptors(mol)
        t2 = get_all_3D_descriptors(mol)
        t3 = get_all_3D_descriptors(mol)
        for i in range(len(t1)):
            self.assertEqual(t1[i], t2[i])
            self.assertEqual(t1[i], t3[i])

    def valid_3D_featurizers_with_nan(self, method, **kwargs):
        dataset_rows_number = len(self.mock_dataset_with_invalid.mols) - 1  # one mol has invalid smiles

        dataset = copy(self.mock_dataset_with_invalid)
        method(**kwargs).featurize(dataset, inplace=True)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

    def test_featurize_with_nan(self):
        self.valid_3D_featurizers_with_nan(All3DDescriptors, mandatory_generation_of_conformers=True)
        self.valid_3D_featurizers_with_nan(AutoCorr3D, mandatory_generation_of_conformers=True)
        self.valid_3D_featurizers_with_nan(RadialDistributionFunction, mandatory_generation_of_conformers=True)
        self.valid_3D_featurizers_with_nan(PlaneOfBestFit, mandatory_generation_of_conformers=True)
        self.valid_3D_featurizers_with_nan(MORSE, mandatory_generation_of_conformers=True)
        self.valid_3D_featurizers_with_nan(WHIM, mandatory_generation_of_conformers=True)
        self.valid_3D_featurizers_with_nan(RadiusOfGyration, mandatory_generation_of_conformers=True)
        self.valid_3D_featurizers_with_nan(InertialShapeFactor, mandatory_generation_of_conformers=True)
        self.valid_3D_featurizers_with_nan(Eccentricity, mandatory_generation_of_conformers=True)
        self.valid_3D_featurizers_with_nan(Asphericity, mandatory_generation_of_conformers=True)
        self.valid_3D_featurizers_with_nan(SpherocityIndex, mandatory_generation_of_conformers=True)
        self.valid_3D_featurizers_with_nan(PrincipalMomentsOfInertia, mandatory_generation_of_conformers=True)
        self.valid_3D_featurizers_with_nan(NormalizedPrincipalMomentsRatios, mandatory_generation_of_conformers=True)

    def test_generate_conformers_and_export(self):
        generate_conformers_to_sdf_file(self.mock_dataset, "temp.sdf", n_conformations=1, max_iterations=1)
        os.remove("temp.sdf")
