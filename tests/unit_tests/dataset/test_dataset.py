import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd

from deepmol.datasets import NumpyDataset


class TestDataset(TestCase):

    def setUp(self) -> None:
        self.mols = ['C', 'CC', 'CCC']
        self.base_dataset = NumpyDataset(mols=self.mols)
        self.output_dir = 'outputs/'
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_numpy_dataset_args(self):
        for i, mol in enumerate(self.mols):
            self.assertIn(mol, self.base_dataset.mols[i])
        self.assertEqual(self.base_dataset.__len__(), 3)

        self.base_dataset.mols = ['C', 'CC', 'CCC', 'CCCC']
        self.assertEqual(self.base_dataset.__len__(), 4)

        with self.assertRaises(ValueError):
            self.base_dataset.y = [1, 0, 1]
        self.base_dataset.y = [1, 0, 0, 1]

        n_mols, n_features, n_y = self.base_dataset.get_shape()
        self.assertEqual(n_mols, (4,))
        self.assertEqual(n_y, (4,))
        self.assertIsNone(n_features)

        self.base_dataset.n_tasks = 1
        self.assertEqual(self.base_dataset.n_tasks, 1)

        self.base_dataset.X = []
        self.assertIsNone(self.base_dataset.X)

        self.base_dataset.ids = None
        self.assertEqual(len(self.base_dataset.ids), len(self.base_dataset.mols))

        with self.assertRaises(ValueError):
            self.base_dataset.ids = [1, 2]
        with self.assertRaises(ValueError):
            self.base_dataset.ids = [1, 1, 3, 4]

    def test_remove_duplicates(self):
        smiles = ['C', 'CC', 'CCC']
        features = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        y = [1, 0, 1]
        ids = [1, 2, 3]
        dataset = NumpyDataset(mols=smiles, X=features, y=y, ids=ids, features2keep=[0, 1, 2], n_tasks=1)
        dataset.remove_duplicates()
        self.assertEqual(len(dataset), 2)

    def test_remove_elements(self):
        smiles = ['C', 'CC', 'CCC', 'CCCC', 'CCCCC']
        ids = [1, 2, 3, 'four', 'five']
        dataset = NumpyDataset(mols=smiles, ids=ids)
        dataset.remove_elements([1, 'five'])
        self.assertEqual(len(dataset), 3)

        dataset.remove_elements(['five'])
        self.assertEqual(len(dataset), 3)

        dataset.remove_elements([2, 3])
        self.assertEqual(len(dataset), 1)

        dataset.remove_elements(['four'])
        self.assertEqual(len(dataset), 0)

    def test_select_features(self):
        smiles = ['C', 'CC', 'CCC']
        features = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        dataset = NumpyDataset(mols=smiles, X=features)
        dataset.select_features([0, 2])
        self.assertEqual(dataset.X.shape[1], 2)
        self.assertEqual(len(dataset.X[0]), 2)
        self.assertEqual(len(dataset.X[1]), 2)
        self.assertEqual(len(dataset.X[2]), 2)

        dataset.select_features(['a', 'c'])
        self.assertEqual(dataset.X.shape[1], 0)

    def test_remove_nan(self):
        smiles = ['C', 'CC', 'CCC', 'CCCC', 'CCCCC']
        features = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
        dataset = NumpyDataset(mols=smiles, X=features)
        dataset.remove_nan()
        dataset.remove_nan(axis=1)
        self.assertEqual(len(dataset), 5)

        dataset.X = [[np.nan, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
        dataset.remove_nan()
        dataset.remove_nan(axis=1)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(dataset.ids), 4)

        dataset.X = [[np.nan, 1.0, np.nan],
                     [np.nan, 1.0, 0.0],
                     [np.nan, 0.0, 1.0],
                     [np.nan, 0.0, 1.0],
                     [np.nan, 0.0, 1.0]]
        dataset.ids = [1, 2, 3, 4]
        dataset.remove_nan(axis=1)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(dataset.ids), 4)
        self.assertEqual(dataset.X.shape[1], 1)
        dataset.remove_nan()
        self.assertEqual(len(dataset), 4)

        dataset.mols = ['C', 'CC', 'CCC']
        dataset.X = [np.nan, 1.0, np.nan]
        dataset.ids = [1, 2, 3]
        dataset.remove_nan()
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.get_shape()[0], (1,))
        self.assertEqual(dataset.get_shape()[1], (1,))
        self.assertIsNone(dataset.get_shape()[2])

    def test_select_to_split(self):
        dataset = NumpyDataset(mols=['C', 'CC', 'CCC', 'CCCC', 'CCCCC'],
                               X=[[1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1]],
                               y=[1, 0, 1, 0, 1],
                               ids=[1, 2, 3, 4, 5],
                               features2keep=[0, 1, 2],
                               n_tasks=1)
        split = dataset.select_to_split([1, 2, 3])
        self.assertEqual(len(split), 3)
        self.assertEqual(len(split.mols), 3)
        self.assertEqual(split.X.shape, (3, 3))
        self.assertEqual(len(split.y), 3)
        self.assertEqual(len(split.ids), 3)
        self.assertEqual(split.n_tasks, 1)

        dataset2 = NumpyDataset(mols=['C', 'CC', 'CCC', 'CCCC', 'CCCCC'],
                                X=[1, 0, 1, 1, 1],
                                y=[1, 0, 1, 0, 1],
                                ids=[1, 2, 3, 4, 5],
                                n_tasks=1)
        split2 = dataset2.select_to_split([1, 2, 3])
        self.assertEqual(len(split2), 3)
        self.assertEqual(len(split2.mols), 3)
        self.assertEqual(split2.X.shape, (3,))
        self.assertEqual(len(split2.y), 3)
        self.assertEqual(len(split2.ids), 3)
        self.assertEqual(split2.n_tasks, 1)

    def test_merge(self):
        d1 = NumpyDataset(mols=['CCCCCCCCCC', 'CCCCCCCCCCCCCCC'],
                          X=[[1, 0, 1], [0, 1, 0]],
                          y=[1, 0],
                          ids=[3, 4])
        d2 = NumpyDataset(mols=['CCCCCCCC', 'CCCCCCCCCCCCC'],
                          X=[[1, 0, 1], [0, 1, 0]],
                          y=[1, 0],
                          ids=[5, 6])

        m = self.base_dataset.merge([d1, d2])
        self.assertIsNone(m.X)
        self.assertEqual(len(m), 7)
        self.assertEqual(len(m.ids), 7)
        self.assertEqual(len(m.y), 7)

        d = NumpyDataset(mols=['CCCCCCCC', 'CCCCCCCCCCCCC', 'c'],
                         X=[[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                         y=[1, 0, 1],
                         ids=[5, 6, 7])
        with self.assertRaises(ValueError):
            d.merge([d1, d2])
        d.ids = [10, 20, 30]
        merged = d.merge([d1, d2])
        self.assertEqual(len(merged), 7)
        self.assertEqual(len(merged), len(d) + len(d1) + len(d2))
        self.assertEqual(len(merged.mols), 7)
        self.assertEqual(len(merged.mols), len(d.mols) + len(d1.mols) + len(d2.mols))
        self.assertEqual(merged.X.shape, (7, 3))
        self.assertEqual(merged.X.shape, (d.X.shape[0] + d1.X.shape[0] + d2.X.shape[0], d.X.shape[1]))

        d3 = NumpyDataset(mols=['CCCCCCCC', 'CCCCCCCCCCCCC', 'c'],
                          X=[1, 0, 1],
                          y=[1, 0, 1],
                          ids=[5, 6, 7])
        d4 = NumpyDataset(mols=['CCCCCCCCCC', 'CCCCCCCCCCCCCCC'],
                          X=[1, 0])
        merged2 = d3.merge([d4])
        self.assertEqual(len(merged2), 5)

        dd = NumpyDataset(mols=['CCCCCCCC', 'CCCCCCCCCCCCC', 'c'],
                          X=[[1, 1], [0, 0], [1, 1]],
                          y=[1, 0, 1],
                          ids=[50, 60, 70])

        merged_x = dd.merge([d4])
        self.assertIsNone(merged_x.X)
        merged_x2 = dd.merge([d])
        self.assertIsNone(merged_x2.X)

    def test_save_to_csv(self):
        df = NumpyDataset(mols=['C', 'CC', 'CCC', 'CCCC', 'CCCCC'],
                          X=[[1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1]],
                          y=[1, 0, 1, 0, 1],
                          ids=[1, 2, 3, 4, 5],
                          features2keep=[0, 1, 2],
                          n_tasks=1)
        df.to_csv(os.path.join(self.output_dir, 'test.csv'))
        pd_df = pd.read_csv(os.path.join(self.output_dir, 'test.csv'))
        # ids + mols + y + features
        self.assertEqual(pd_df.shape, (len(df.mols), 3 + df.X.shape[1]))

    def test_save_load_features(self):
        d = NumpyDataset(mols=['C', 'CC', 'CCC', 'CCCC', 'CCCCC'])
        with self.assertRaises(ValueError):
            d.save_features(os.path.join(self.output_dir, 'test.csv'))

        d1 = NumpyDataset(mols=['CCCCCCCCCC', 'CCCCCCCCCCCCCCC'],
                          X=[[1, 0, 1], [0, 1, 0]],
                          y=[1, 0],
                          ids=[3, 4])
        d1.save_features(os.path.join(self.output_dir, 'test.csv'))

        d2 = NumpyDataset(mols=['CCCCCCCCCC', 'CCCCCCCCCCCCCCC'])
        d2.load_features(os.path.join(self.output_dir, 'test.csv'))

        self.assertEqual(d1.X.shape, d2.X.shape)
        for i in range(d1.X.shape[0]):
            for j in range(d1.X.shape[1]):
                self.assertEqual(d1.X[i, j], d2.X[i, j])
