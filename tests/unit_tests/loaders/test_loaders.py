import os
from unittest import TestCase

from deepmol.loaders import SDFLoader, CSVLoader
from tests import TEST_DIR


class TestLoaders(TestCase):

    def test_sdf_loader(self) -> None:
        self.data_path = os.path.join(TEST_DIR, 'data')
        dataset = os.path.join(self.data_path, "results_test.sdf")
        loader = SDFLoader(dataset)
        sdf_dataset = loader.create_dataset()
        self.assertEqual(len(sdf_dataset.mols), 5)

        loader2 = SDFLoader(dataset, id_field='_ID', labels_fields='_Class', shard_size=2)
        dataset2 = loader2.create_dataset()
        self.assertEqual(len(dataset2.mols), 2)
        self.assertIsNone(dataset2.X)
        self.assertEqual(len(dataset2.y), 2)

        loader3 = SDFLoader(dataset,
                            id_field='_ID',
                            labels_fields=['_Class', '_Class2'],
                            features_fields='_Feature1',
                            shard_size=2)
        dataset3 = loader3.create_dataset()
        self.assertEqual(len(dataset3.mols), 2)
        self.assertEqual(dataset3.X.shape, (2, 1))
        self.assertEqual(dataset3.y.shape, (2, 2))

        loader4 = SDFLoader(dataset,
                            id_field='_ID',
                            labels_fields=['_Class'],
                            features_fields=['_Feature1', '_Feature2'],
                            shard_size=3)
        dataset4 = loader4.create_dataset()
        self.assertEqual(len(dataset4.mols), 3)
        self.assertEqual(dataset4.X.shape, (3, 2))
        self.assertEqual(len(dataset4.y), 3)

    def test_csv_loader(self):
        data_path = os.path.join(TEST_DIR, 'data')
        dataset_path = os.path.join(data_path, "train_dataset.csv")

        csv1 = CSVLoader(dataset_path,
                         mols_field='mols',
                         labels_fields='y',
                         id_field='ids',
                         features_fields=['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5',
                                          'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10'],
                         shard_size=10)
        df1 = csv1.create_dataset()
        self.assertEqual(len(df1.mols), 10)
        self.assertEqual(df1.X.shape, (10, 10))

        data_path2 = os.path.join(TEST_DIR, 'data')
        dataset_path2 = os.path.join(data_path2, "balanced_mini_dataset.csv")
        csv2 = CSVLoader(dataset_path2, mols_field='Smiles')
        df2 = csv2.create_dataset(sep=';')
        self.assertEqual(len(df2.mols), 14)

        csv3 = CSVLoader(dataset_path,
                         mols_field='mols',
                         labels_fields=['y', 'feat_1024'],
                         features_fields='feat_1',
                         shard_size=10)
        df3 = csv3.create_dataset()
        self.assertEqual(len(df3.mols), 10)
        self.assertEqual(df3.y.shape, (10, 2))
