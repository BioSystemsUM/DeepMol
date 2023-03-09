import os
from abc import ABC, abstractmethod
from unittest.mock import MagicMock

import numpy as np
from rdkit.Chem import MolFromSmiles

from deepmol.datasets import SmilesDataset


class ModelsTestCase(ABC):

    def setUp(self) -> None:
        # create 100*50 binary numpy array
        x = np.random.randint(2, size=(100, 50))
        x_test = np.random.randint(2, size=(10, 50))
        # binary labels
        y = np.random.randint(2, size=(100,))
        y_test = np.random.randint(2, size=(10,))
        # multiclass labels
        y_multiclass = np.random.randint(3, size=(100,))
        y_multiclass_test = np.random.randint(3, size=(10,))
        # multitask labels
        y_multitask = np.random.randint(2, size=(100, 3))
        y_multitask_test = np.random.randint(2, size=(10, 3))
        # regression labels between 0 and 1
        y_regression = np.random.rand(100, )
        y_regression_test = np.random.rand(10, )
        # multitask regression labels between 0 and 1
        y_multitask_regression = np.random.rand(100, 3)
        y_multitask_regression_test = np.random.rand(10, 3)

        ids = [str(i) for i in range(100)]
        ids_test = [str(i) for i in range(100, 110)]
        # create binary classification dataset
        self.binary_dataset = MagicMock(spec=SmilesDataset,
                                        X=x,
                                        y=y,
                                        n_tasks=1,
                                        label_names=['binary_label'],
                                        mode='classification',
                                        ids=ids)
        self.binary_dataset.__len__.return_value = 100
        self.binary_dataset.X = x
        self.binary_dataset.y = y
        self.binary_dataset.n_tasks = 1
        self.binary_dataset.label_names = ['binary_label']
        self.binary_dataset.ids = ids
        self.binary_dataset_test = MagicMock(spec=SmilesDataset,
                                             X=x_test,
                                             y=y_test,
                                             n_tasks=1,
                                             label_names=['binary_label'],
                                             mode='classification',
                                             ids=ids_test)
        self.binary_dataset_test.__len__.return_value = 10
        self.binary_dataset_test.X = x_test
        self.binary_dataset_test.y = y_test
        self.binary_dataset_test.n_tasks = 1
        self.binary_dataset_test.label_names = ['binary_label']
        self.binary_dataset_test.ids = ids_test

        # create multiclass classification dataset
        self.multiclass_dataset = MagicMock(spec=SmilesDataset,
                                            X=x,
                                            y=y_multiclass,
                                            n_tasks=1,
                                            label_names=['multiclass_label'])
        self.multiclass_dataset.__len__.return_value = 100
        self.multiclass_dataset.X = x
        self.multiclass_dataset.y = y_multiclass
        self.multiclass_dataset.n_tasks = 1
        self.multiclass_dataset.label_names = ['multiclass_label']
        self.multiclass_dataset_test = MagicMock(spec=SmilesDataset,
                                                 X=x_test,
                                                 y=y_multiclass_test,
                                                 n_tasks=1,
                                                 label_names=['multiclass_label'])
        self.multiclass_dataset_test.__len__.return_value = 10
        self.multiclass_dataset_test.X = x_test
        self.multiclass_dataset_test.y = y_multiclass_test
        self.multiclass_dataset_test.n_tasks = 1
        self.multiclass_dataset_test.label_names = ['multiclass_label']

        # create multitask classification dataset
        self.multitask_dataset = MagicMock(spec=SmilesDataset,
                                           X=x,
                                           y=y_multitask,
                                           ids=ids,
                                           n_tasks=3,
                                           label_names=['multitask_label_1', 'multitask_label_2', 'multitask_label_3'],
                                           mode='multitask')
        self.multitask_dataset.__len__.return_value = 100
        self.multitask_dataset.X = x
        self.multitask_dataset.y = y_multitask
        self.multitask_dataset.n_tasks = 3
        self.multitask_dataset.label_names = ['multitask_label_1', 'multitask_label_2', 'multitask_label_3']
        self.multitask_dataset.ids = ids
        self.multitask_dataset_test = MagicMock(spec=SmilesDataset,
                                                X=x_test,
                                                y=y_multitask_test,
                                                ids=ids_test,
                                                n_tasks=3,
                                                label_names=['multitask_label_1', 'multitask_label_2',
                                                             'multitask_label_3'],
                                                mode='multitask')
        self.multitask_dataset_test.__len__.return_value = 10
        self.multitask_dataset_test.X = x_test
        self.multitask_dataset_test.y = y_multitask_test
        self.multitask_dataset_test.n_tasks = 3
        self.multitask_dataset_test.label_names = ['multitask_label_1', 'multitask_label_2', 'multitask_label_3']
        self.multitask_dataset_test.ids = ids_test

        # create regression dataset
        self.regression_dataset = MagicMock(spec=SmilesDataset,
                                            X=x,
                                            y=y_regression,
                                            n_tasks=1,
                                            label_names=['regression_label'])
        self.regression_dataset.__len__.return_value = 100
        self.regression_dataset.X = x
        self.regression_dataset.y = y_regression
        self.regression_dataset.n_tasks = 1
        self.regression_dataset.label_names = ['regression_label']
        self.regression_dataset_test = MagicMock(spec=SmilesDataset,
                                                 X=x_test,
                                                 y=y_regression_test,
                                                 n_tasks=1,
                                                 label_names=['regression_label'])
        self.regression_dataset_test.__len__.return_value = 100
        self.regression_dataset_test.X = x_test
        self.regression_dataset_test.y = y_regression_test
        self.regression_dataset_test.n_tasks = 1
        self.regression_dataset_test.label_names = ['regression_label']

        # create multitask regression dataset
        self.multitask_regression_dataset = MagicMock(spec=SmilesDataset,
                                                      X=x,
                                                      y=y_multitask_regression,
                                                      n_tasks=3,
                                                      label_names=['multitask_regression_label_1',
                                                                   'multitask_regression_label_2',
                                                                   'multitask_regression_label_3'])
        self.multitask_regression_dataset.__len__.return_value = 100
        self.multitask_regression_dataset.X = x
        self.multitask_regression_dataset.y = y_multitask_regression
        self.multitask_regression_dataset.n_tasks = 3
        self.multitask_regression_dataset.label_names = ['multitask_regression_label_1',
                                                         'multitask_regression_label_2',
                                                         'multitask_regression_label_3']
        self.multitask_regression_dataset_test = MagicMock(spec=SmilesDataset,
                                                           X=x_test,
                                                           y=y_multitask_regression_test,
                                                           n_tasks=3,
                                                           label_names=['multitask_regression_label_1',
                                                                        'multitask_regression_label_2',
                                                                        'multitask_regression_label_3'])
        self.multitask_regression_dataset_test.__len__.return_value = 10
        self.multitask_regression_dataset_test.X = x_test
        self.multitask_regression_dataset_test.y = y_multitask_regression_test
        self.multitask_regression_dataset_test.n_tasks = 3
        self.multitask_regression_dataset_test.label_names = ['multitask_regression_label_1',
                                                              'multitask_regression_label_2',
                                                              'multitask_regression_label_3']

    def tearDown(self) -> None:
        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')

    @abstractmethod
    def test_fit_predict_evaluate(self):
        raise NotImplementedError
