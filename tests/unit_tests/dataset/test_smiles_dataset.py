import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd
from rdkit.Chem import Mol, MolFromSmiles

from deepmol.datasets import SmilesDataset
from deepmol.loaders._utils import load_sdf_file
from deepmol.loggers import Logger
from deepmol.models._utils import get_prediction_from_proba


class TestSmilesDataset(TestCase):

    def setUp(self) -> None:
        self.smiles = ['C', 'CC', 'CCC']
        self.base_dataset = SmilesDataset(smiles=self.smiles)
        self.output_dir = 'outputs/'
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        # Close logger file handlers to release the file
        singleton_instance = Logger()
        singleton_instance.close_handlers()
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')

    def test_infer_mode(self):
        df3 = SmilesDataset(smiles=['C', 'CC', 'CCC'], y=[[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        assert df3.mode == ['classification', 'classification', 'classification']

        df3 = SmilesDataset(smiles=['C', 'CC', 'CCC'], y=[[3, 1, 0], [2, 1, 0], [1, 0, 1]])
        assert df3.mode == ['classification', 'classification', 'classification']

        df3 = SmilesDataset(smiles=['C', 'CC', 'CCC'], y=[[3.2, 1, 0], [2, 1, 0], [1, 0, 1]])
        assert df3.mode == ['regression', 'classification', 'classification']

        df3 = SmilesDataset(smiles=['C', 'CC', 'CCC'], y=[3.2, 3.1, 3.0])
        assert df3.mode == 'regression'

        df3 = SmilesDataset(smiles=['C', 'CC', 'CCC'], y=[3., 3., 3.])
        assert df3.mode == 'classification'

    def test_utils(self):

        dataset = SmilesDataset(smiles=['C', 'CC', 'CCC'], y=[[1, 0, 0], [0, 1, 0], [1, 0, 1]])
        y_pred_proba = np.array([[0.2, 0.1, 0], [0.3, 0.7, 0], [0.99, 0.2, 1]])
        prediction = get_prediction_from_proba(dataset, y_pred_proba)
        assert np.array_equal(prediction, np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1]]))

        dataset = SmilesDataset(smiles=['C', 'CC', 'CCC'], y=[0, 1, 2])
        y_pred_proba = np.array([[0.7, 0.1, 0.2], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
        prediction = get_prediction_from_proba(dataset, y_pred_proba)
        assert np.array_equal(prediction, np.array([0, 1, 2]))

        dataset = SmilesDataset(smiles=['C', 'CC', 'CCC'], y=[[0, 1, 3.2], [1, 1, 3.2], [1, 0, 3.2]])
        y_pred_proba = np.array([[0.7, 0.6, 3.2], [0.1, 1, 3.2], [0.2, 0.1, 3.2]])
        prediction = get_prediction_from_proba(dataset, y_pred_proba)
        assert np.array_equal(prediction, np.array([[1, 1, 3.2], [0, 1, 3.2], [0, 0, 3.2]]))

    def test_smiles_dataset_args(self):
        for i, mol in enumerate(self.smiles):
            self.assertEqual(mol, self.base_dataset.smiles[i])
        self.assertEqual(self.base_dataset.__len__(), 3)
        self.assertFalse(self.base_dataset.ids is None)
        self.assertTrue(self.base_dataset.X is None)
        self.assertTrue(self.base_dataset.y is None)
        self.assertEqual(self.base_dataset.n_tasks, 0)
        for mol in self.base_dataset.mols:
            self.assertIsInstance(mol, Mol)
        smiles_shape, x_shape, y_shape = self.base_dataset.get_shape()
        self.assertEqual(smiles_shape, (3,))
        self.assertEqual(x_shape, None)
        self.assertEqual(y_shape, None)

        self.base_dataset.smiles = ['C', 'CC', 'CCC', 'CCCC', 'CCCCC(']
        self.assertEqual(self.base_dataset.__len__(), 4)
        with self.assertRaises(AttributeError):
            self.base_dataset.mols = [MolFromSmiles('C'), MolFromSmiles('CC')]
        with self.assertRaises(AttributeError):
            self.base_dataset.X = [1, 2, 3, 4]
        with self.assertRaises(AttributeError):
            self.base_dataset.y = [1, 0, 1, 0]
        with self.assertRaises(AttributeError):
            self.base_dataset.n_tasks = 1
        with self.assertRaises(ValueError):
            self.base_dataset.ids = [1, 2]
        with self.assertRaises(ValueError):
            self.base_dataset.ids = [1, 1, 3, 4]
        self.base_dataset.ids = [1, 2, 3, 4]

        with self.assertRaises(ValueError):
            self.base_dataset.select(ids=[1, 2, 3, 4, 5], axis=3)

        with self.assertRaises(ValueError):
            self.base_dataset.feature_names = ['feature1', 'feature2']

    def test_get_n_tasks(self):
        df1 = SmilesDataset(smiles=['C', 'CC', 'CCC'], y=[1, 0, 1])
        self.assertEqual(df1.n_tasks, 1)
        df2 = SmilesDataset(smiles=['C', 'CC', 'CCC'], y=[[1, 0], [0, 1], [1, 0]])
        self.assertEqual(df2.n_tasks, 2)
        df3 = SmilesDataset(smiles=['C', 'CC', 'CCC'], y=[[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        self.assertEqual(df3.n_tasks, 3)

    def test_remove_duplicates(self):
        df0 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'])
        self.assertEqual(df0.__len__(), 7)
        df0.remove_duplicates(inplace=True)
        self.assertEqual(df0.__len__(), 7)

        df = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                           X=[1, 1, 3, 4, 5, 6, 1])
        self.assertEqual(df.__len__(), 7)
        df.remove_duplicates(inplace=True)
        self.assertEqual(df.__len__(), 5)
        self.assertTrue('CC' not in df.smiles)
        self.assertTrue('CCCCC' not in df.smiles)

        df2 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                            X=[[1, 1], [1, 1], [3, 4], [4, 5], [5, 6], [6, 1], [1, 1]])
        self.assertEqual(df2.__len__(), 7)
        df2.remove_duplicates(inplace=True)
        self.assertEqual(df2.__len__(), 5)
        self.assertTrue('CC' not in df2.smiles)
        self.assertTrue('CCCCC' not in df2.smiles)

        df3 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                            X=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
                               [[1, 0, 1], [0, 1, 0], [0, 0, 1]],
                               [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
                               [[1, 0, 0], [0, 1, 1], [0, 0, 1]],
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        self.assertEqual(df3.__len__(), 7)
        df3.remove_duplicates(inplace=True)
        self.assertEqual(df3.__len__(), 5)
        self.assertTrue('CC' not in df3.smiles)
        self.assertTrue('CCCCC' not in df3.smiles)

        df2 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                            X=[[1, 1], [1, np.nan], [3, 4], [4, 5], [5, 6], [6, 1], [1, np.nan]])
        self.assertEqual(df2.__len__(), 7)
        df2.remove_duplicates(inplace=True)
        self.assertEqual(df2.__len__(), 7)

        df2.feature_names = ['feature1', 'feature2']
        self.assertEqual(df2.feature_names[0], 'feature1')
        self.assertEqual(df2.feature_names[1], 'feature2')
        with self.assertRaises(ValueError):
            df2.feature_names = ['feature1', 'feature2', 'feature3']
        with self.assertRaises(ValueError):
            df2.feature_names = ['feature1', 'feature1']

        # inplace=False
        df0 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                            X=[1, 1, 3, 4, 5, 6, 1])
        self.assertEqual(df0.__len__(), 7)
        df0.remove_duplicates()
        self.assertEqual(df0.__len__(), 7)
        df00 = df0.remove_duplicates(inplace=False)
        self.assertEqual(df00.__len__(), 5)
        self.assertEqual(df0.__len__(), 7)

    def test_remove_elements(self):
        df0 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                            X=[1, 1, 3, 4, 5, 6, 1],
                            ids=[0,1,2,3,4,5,6])
        
        self.assertEqual(df0.__len__(), 7)
        df0.remove_elements(ids=["0","1","2"])
        self.assertEqual(df0.__len__(), 7)
        self.assertEqual(df0.removed_elements, [])
        df00 = df0.remove_elements(ids=["0","1","2"], inplace=False)
        self.assertEqual(df00.__len__(), 4)
        self.assertEqual(df00.removed_elements, [0,1,2])
        self.assertEqual(df0.__len__(), 7)
        self.assertEqual(df0.removed_elements, [])
        df0.remove_elements(ids=["0","1","2"], inplace=True)
        self.assertEqual(df0.removed_elements, [0,1,2])

    def test_remove_nan_axis_0(self):
        df0 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'])
        self.assertEqual(df0.__len__(), 7)
        df0.remove_nan(inplace=True)
        self.assertEqual(df0.__len__(), 7)

        df = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                           X=[1, np.nan, 3, 4, 5, 6, 1])
        self.assertEqual(df.__len__(), 7)
        df.remove_nan()
        self.assertEqual(df.__len__(), 7)
        self.assertEqual(df.removed_elements, [])
        dff = df.remove_nan()
        self.assertEqual(dff.__len__(), 6)
        self.assertEqual(dff.removed_elements, [1])
        df.remove_nan(inplace=True)
        self.assertEqual(df.__len__(), 6)
        self.assertEqual(df.removed_elements, [1])

        df2 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                            X=[[1, 1], [1, np.nan], [3, 4], [4, 5], [5, 6], [6, 1], [1, np.nan]])
        self.assertEqual(df2.__len__(), 7)
        self.assertEqual(df2.removed_elements, [])
        df2.remove_nan(inplace=True)
        self.assertEqual(df2.__len__(), 5)
        self.assertEqual(df2.removed_elements, [1, 6])

        df3 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                            X=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               [[1, 0, np.nan], [0, 1, 0], [0, 0, 1]],
                               [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
                               [[1, 0, 1], [0, 1, 0], [0, 0, 1]],
                               [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
                               [[1, 0, 0], [0, 1, 1], [0, 0, 1]],
                               [[1, 0, 0], [0, 1, 0], [0, 0, np.nan]]])
        self.assertEqual(df3.__len__(), 7)
        df3.remove_nan(inplace=True)
        self.assertEqual(df3.__len__(), 5)

    def test_remove_nan_axis_1(self):
        df0 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'])
        self.assertEqual(df0.__len__(), 7)
        df0.remove_nan(axis=1, inplace=True)
        self.assertEqual(df0.__len__(), 7)

        df = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                           X=[1, np.nan, 3, 4, 5, 6, 1])
        self.assertEqual(df.__len__(), 7)
        df.remove_nan(axis=1)
        self.assertEqual(df.__len__(), 7)
        dff = df.remove_nan()
        self.assertEqual(dff.__len__(), 6)
        df.remove_nan(inplace=True)
        self.assertEqual(df.__len__(), 6)

        df2 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                            X=[[1, 1], [np.nan, np.nan], [3, 4], [4, 5], [5, 6], [6, 1], [1, np.nan]])
        self.assertEqual(df2.__len__(), 7)
        df2.remove_nan(axis=1, inplace=True)
        self.assertEqual(df2.__len__(), 6)
        self.assertEqual(df2.X.shape, (6, 1))
        self.assertEqual(df2.feature_names, ['feature_0'])

        df3 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                            X=[[1, np.nan],
                               [np.nan, np.nan],
                               [3, np.nan],
                               [4, np.nan],
                               [5, np.nan],
                               [6, np.nan],
                               [1, np.nan]])
        self.assertEqual(df3.__len__(), 7)
        df3.remove_nan(axis=1, inplace=True)
        self.assertEqual(df3.__len__(), 6)
        self.assertEqual(df3.X.shape, (6, 1))
        self.assertEqual(df2.feature_names, ['feature_0'])

        df3 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                            X=[[[1, 0, 0], [0, 1, np.nan], [0, 0, 1]],
                               [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                               [[1, 1, 1], [0, 1, np.nan], [0, 0, 1]],
                               [[1, 0, 0], [0, 1, np.nan], [0, 0, 1]],
                               [[1, 0, 1], [1, 1, np.nan], [0, 0, 1]],
                               [[1, 0, 0], [0, 1, np.nan], [0, 0, 1]],
                               [[1, 0, 1], [0, 1, np.nan], [0, 0, 1]]])
        self.assertEqual(df3.__len__(), 7)
        df3.remove_nan(axis=1, inplace=True)
        self.assertEqual(df3.__len__(), 6)
        self.assertEqual(df3.X.shape, (6, 2, 3))

        with self.assertRaises(ValueError):
            df3.remove_nan(axis=2)

    def test_select_features_by_index(self):
        df0 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'])
        self.assertEqual(df0.X, None)
        self.assertEqual(df0.feature_names, None)
        with self.assertRaises(ValueError):
            df0.select_features_by_index([0])

        df = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                           X=[1, 2, 3, 4, 5, 6, 1],
                           feature_names=['XXX'])
        self.assertEqual(df.X.shape, (7,))
        df.select_features_by_index([0], inplace=True)
        self.assertEqual(df.feature_names, [['XXX']])

        df = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                           X=[[1, 1], [1, 2], [3, 4], [4, 5], [5, 6], [6, 1], [1, 1]],
                           feature_names=['XXX', 'YYY'])
        self.assertEqual(df.X.shape, (7, 2))
        df.select_features_by_index([1])
        self.assertEqual(df.X.shape, (7, 2))
        df0 = df.select_features_by_index([1])
        self.assertEqual(df0.X.shape, (7, 1))
        df.select_features_by_index([1], inplace=True)
        self.assertEqual(df.X.shape, (7, 1))
        self.assertEqual(df.feature_names, ['YYY'])

        df = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC'],
                           X=[[1, 1, 1], [1, 2, 3], [3, 4, 4], [4, 5, 6]],
                           feature_names=['XXX', 'YYY', 'ZZZ'])
        self.assertEqual(df.X.shape, (4, 3))
        df.select_features_by_index([0, 2], inplace=True)
        self.assertEqual(df.X.shape, (4, 2))
        self.assertEqual(df.feature_names, ['XXX', 'ZZZ'])

    def test_select_features_by_name(self):
        df = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                           X=[1, 2, 3, 4, 5, 6, 1],
                           feature_names=['XXX'])
        self.assertEqual(df.X.shape, (7,))
        df.select_features_by_name(['XXX'], inplace=True)
        self.assertEqual(df.feature_names, [['XXX']])

        df = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCC', 'CCCCC'],
                           X=[[1, 1], [1, 2], [3, 4], [4, 5], [5, 6], [6, 1], [1, 1]],
                           feature_names=['XXX', 'YYY'])
        self.assertEqual(df.X.shape, (7, 2))
        df.select_features_by_name(['YYY'])
        self.assertEqual(df.X.shape, (7, 2))
        df0 = df.select_features_by_name(['YYY'])
        self.assertEqual(df0.X.shape, (7, 1))
        df.select_features_by_name(['YYY'], inplace=True)
        self.assertEqual(df.X.shape, (7, 1))
        self.assertEqual(df.feature_names, ['YYY'])

        df = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCC'],
                           X=[[1, 1, 1], [1, 2, 3], [3, 4, 4], [4, 5, 6]],
                           feature_names=['XXX', 'YYY', 'ZZZ'])
        self.assertEqual(df.X.shape, (4, 3))
        df.select_features_by_name(['XXX', 'ZZZ'], inplace=True)
        self.assertEqual(df.X.shape, (4, 2))
        self.assertEqual(df.feature_names, ['XXX', 'ZZZ'])

    def test_select_to_split(self):
        dataset = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCCC', 'CCCCC'],
                                X=[[1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1]],
                                y=[1, 0, 1, 0, 1],
                                ids=[1, 2, 3, 4, 5])
        split = dataset.select_to_split([1, 2, 3])
        self.assertEqual(len(split), 3)
        self.assertEqual(len(split.smiles), 3)
        for i, m in enumerate(split.smiles):
            self.assertEqual(m, ['CC', 'CCC', 'CCCC'][i])
        self.assertEqual(split.X.shape, (3, 3))
        self.assertEqual(len(split.y), 3)
        self.assertEqual(len(split.ids), 3)
        self.assertEqual(split.n_tasks, 1)

        dataset2 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCCC', 'CCCCC'],
                                 X=[1, 0, 1, 1, 1],
                                 y=[1, 0, 1, 0, 1],
                                 ids=[1, 2, 3, 4, 5])
        split2 = dataset2.select_to_split([1, 2, 3])
        self.assertEqual(len(split2), 3)
        self.assertEqual(len(split2.mols), 3)
        self.assertEqual(split2.X.shape, (3,))
        self.assertEqual(len(split2.y), 3)
        self.assertEqual(len(split2.ids), 3)
        self.assertEqual(split2.n_tasks, 1)

    def test_merge(self):
        d1 = SmilesDataset(smiles=['CCCCCCCCCC', 'CCCCCCCCCCCCCCC'],
                           X=[[1, 0, 1], [0, 1, 0]],
                           y=[1, 0],
                           ids=[3, 4])
        d2 = SmilesDataset(smiles=['CCCCCCCC', 'CCCCCCCCCCCCC'],
                           X=[[1, 0, 1], [0, 1, 0]],
                           y=[1, 0],
                           ids=[5, 6])

        m = self.base_dataset.merge([d1, d2])
        self.assertIsNone(m.X)
        self.assertEqual(len(m), 7)
        self.assertEqual(len(m.ids), 7)
        self.assertEqual(len(m.y), 7)

        d = SmilesDataset(smiles=['CCCCCCCC', 'CCCCCCCCCCCCC', 'C'],
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

        d3 = SmilesDataset(smiles=['CCCCCCCC', 'CCCCCCCCCCCCC', 'C'],
                           X=[1, 0, 1],
                           y=[1, 0, 1],
                           ids=[5, 6, 7])
        d4 = SmilesDataset(smiles=['CCCCCCCCCC', 'CCCCCCCCCCCCCCC'],
                           X=[1, 0])
        merged2 = d3.merge([d4])
        self.assertEqual(len(merged2), 5)

        dd = SmilesDataset(smiles=['CCCCCCCC', 'CCCCCCCCCCCCC', 'C'],
                           X=[[1, 1], [0, 0], [1, 1]],
                           y=[1, 0, 1],
                           ids=[50, 60, 70])

        merged_x = dd.merge([d4])
        self.assertIsNone(merged_x.X)
        merged_x2 = dd.merge([d])
        self.assertIsNone(merged_x2.X)

    def test_to_dataframe(self):
        df = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCCC', 'CCCCC'],
                           X=[[1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1]],
                           y=[1, 0, 1, 0, 1],
                           ids=[1, 2, 3, 4, 5],
                           feature_names=['XXX', 'YYY', 'ZZZ'])
        
        pd_df = df.to_dataframe()
        self.assertEqual(pd_df.shape, (len(df.mols), 3 + df.X.shape[1]))
        for i, col_name in enumerate(pd_df.columns.values):
            self.assertEqual(col_name, ['ids', 'smiles', 'y', 'XXX', 'YYY', 'ZZZ'][i])

    def test_save_to_csv(self):
        df = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCCC', 'CCCCC'],
                           X=[[1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1]],
                           y=[1, 0, 1, 0, 1],
                           ids=[1, 2, 3, 4, 5],
                           feature_names=['XXX', 'YYY', 'ZZZ'])
        df.to_csv(os.path.join(self.output_dir, 'test.csv'), index=False, sep=",")
        pd_df = pd.read_csv(os.path.join(self.output_dir, 'test.csv'))
        # ids + mols + y + features
        self.assertEqual(pd_df.shape, (len(df.mols), 3 + df.X.shape[1]))
        for i, col_name in enumerate(pd_df.columns.values):
            self.assertEqual(col_name, ['ids', 'smiles', 'y', 'XXX', 'YYY', 'ZZZ'][i])

    def test_save_to_sdf(self):
        dataset = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCCC', 'CCCCC'],
                                X=[[1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1]],
                                y=[1, 0, 1, 0, 1],
                                ids=[1, 2, 3, 4, 5])
        dataset.to_sdf(os.path.join(self.output_dir, 'test.sdf'))
        mols = load_sdf_file(os.path.join(self.output_dir, 'test.sdf'))
        y = [1, 0, 1, 0, 1]
        for i, mol in enumerate(mols):
            y_i = mol.GetProp("y")
            self.assertEqual(y[i], int(float(y_i)))

        dataset2 = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCCC', 'CCCCC'],
                                 X=[1, 0, 1, 1, 1],
                                 y=np.array([[1, 0, 1, 0, 1], [1, 0, 1, 0, 1]]).T,
                                 ids=[1, 2, 3, 4, 5])
        dataset2.to_sdf(os.path.join(self.output_dir, 'test.sdf'))
        mols = load_sdf_file(os.path.join(self.output_dir, 'test.sdf'))
        y = np.array([[1, 0, 1, 0, 1], [1, 0, 1, 0, 1]]).T
        for i, mol in enumerate(mols):
            y_i = mol.GetProp("y_0")
            self.assertEqual(y[i, 0], int(float(y_i)))
            y_i = mol.GetProp("y_1")
            self.assertEqual(y[i, 1], int(float(y_i)))


    def test_save_load_features(self):
        d = SmilesDataset(smiles=['C', 'CC', 'CCC', 'CCCC', 'CCCCC'])
        with self.assertRaises(ValueError):
            d.save_features(os.path.join(self.output_dir, 'test.csv'))

        d1 = SmilesDataset(smiles=['CCCCCCCCCC', 'CCCCCCCCCCCCCCC'],
                           X=[[1, 0, 1], [0, 1, 0]],
                           y=[1, 0],
                           ids=[3, 4],
                           feature_names=['XXX', 'YYY', 'ZZZ'])
        d1.save_features(os.path.join(self.output_dir, 'test.csv'))

        d2 = SmilesDataset(smiles=['CCCCCCCCCC', 'CCCCCCCCCCCCCCC'])
        d2.load_features(os.path.join(self.output_dir, 'test.csv'))

        with self.assertRaises(AttributeError):
            self.assertEqual(d1.X.shape, d2.X.shape)
        d3 = d2.load_features(os.path.join(self.output_dir, 'test.csv'))
        self.assertEqual(d1.X.shape, d3.X.shape)
        for i in range(d1.X.shape[0]):
            for j in range(d1.X.shape[1]):
                self.assertEqual(d1.X[i, j], d3.X[i, j])

        d2.load_features(os.path.join(self.output_dir, 'test.csv'), inplace=True)
        self.assertEqual(d1.X.shape, d2.X.shape)
        for i in range(d1.X.shape[0]):
            for j in range(d1.X.shape[1]):
                self.assertEqual(d1.X[i, j], d2.X[i, j])
