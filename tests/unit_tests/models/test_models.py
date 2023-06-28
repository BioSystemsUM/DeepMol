import os
from abc import ABC, abstractmethod
from unittest.mock import MagicMock

import numpy as np
from rdkit.Chem import MolFromSmiles

from deepmol.datasets import SmilesDataset
from deepmol.models._utils import get_prediction_from_proba
from unit_tests._mock_utils import SmilesDatasetMagicMock


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

        ids = np.array([str(i) for i in range(100)])
        ids_test = np.array([str(i) for i in range(100, 110)])
        # create binary classification dataset
        smiles = ['CCC' * 50 for _ in range(100)]
        mols = [MolFromSmiles(smi) for smi in smiles]
        self.binary_dataset = SmilesDatasetMagicMock(spec=SmilesDataset,
                                                     X=x,
                                                     y=y,
                                                     n_tasks=1,
                                                     label_names=['binary_label'],
                                                     mode='classification',
                                                     ids=ids,
                                                     smiles=smiles)
        self.binary_dataset.mols = [MolFromSmiles(smi) for smi in smiles]
        self.binary_dataset.select_to_split.side_effect = lambda arg: MagicMock(spec=SmilesDataset,
                                                                                X=x[arg],
                                                                                y=y[arg],
                                                                                n_tasks=1,
                                                                                label_names=['binary_label'],
                                                                                mode='classification',
                                                                                ids=ids[arg])
        self.binary_dataset.__len__.return_value = 100
        self.binary_dataset_test = SmilesDatasetMagicMock(spec=SmilesDataset,
                                                          X=x_test,
                                                          y=y_test,
                                                          n_tasks=1,
                                                          label_names=['binary_label'],
                                                          mode='classification',
                                                          ids=ids_test)
        self.binary_dataset_test.__len__.return_value = 10

        # create multiclass classification dataset
        self.multiclass_dataset = SmilesDatasetMagicMock(spec=SmilesDataset,
                                                         X=x,
                                                         y=y_multiclass,
                                                         n_tasks=1,
                                                         label_names=['multiclass_label'])
        self.multiclass_dataset.__len__.return_value = 100
        self.multiclass_dataset_test = SmilesDatasetMagicMock(spec=SmilesDataset,
                                                              X=x_test,
                                                              y=y_multiclass_test,
                                                              n_tasks=1,
                                                              label_names=['multiclass_label'])
        self.multiclass_dataset_test.__len__.return_value = 10

        # create multitask classification dataset
        self.multitask_dataset = SmilesDatasetMagicMock(spec=SmilesDataset,
                                                        X=x,
                                                        y=y_multitask,
                                                        ids=ids,
                                                        n_tasks=3,
                                                        label_names=['multitask_label_1', 'multitask_label_2',
                                                                     'multitask_label_3'],
                                                        mode=["classification"] * 3)
        self.multitask_dataset.__len__.return_value = 100
        self.multitask_dataset_test = SmilesDatasetMagicMock(spec=SmilesDataset,
                                                             X=x_test,
                                                             y=y_multitask_test,
                                                             ids=ids_test,
                                                             n_tasks=3,
                                                             label_names=['multitask_label_1', 'multitask_label_2',
                                                                          'multitask_label_3'],
                                                             mode=["classification"] * 3)
        self.multitask_dataset_test.__len__.return_value = 10

        # create regression dataset
        self.regression_dataset = SmilesDatasetMagicMock(spec=SmilesDataset,
                                                         X=x,
                                                         y=y_regression,
                                                         n_tasks=1,
                                                         label_names=['regression_label'])
        self.regression_dataset.__len__.return_value = 100
        self.regression_dataset_test = SmilesDatasetMagicMock(spec=SmilesDataset,
                                                              X=x_test,
                                                              y=y_regression_test,
                                                              n_tasks=1,
                                                              label_names=['regression_label'])
        self.regression_dataset_test.__len__.return_value = 100

        # create multitask regression dataset
        self.multitask_regression_dataset = SmilesDatasetMagicMock(spec=SmilesDataset,
                                                                   X=x,
                                                                   y=y_multitask_regression,
                                                                   n_tasks=3,
                                                                   label_names=['multitask_regression_label_1',
                                                                                'multitask_regression_label_2',
                                                                                'multitask_regression_label_3'])
        self.multitask_regression_dataset.__len__.return_value = 100
        self.multitask_regression_dataset_test = SmilesDatasetMagicMock(spec=SmilesDataset,
                                                                        X=x_test,
                                                                        y=y_multitask_regression_test,
                                                                        n_tasks=3,
                                                                        label_names=['multitask_regression_label_1',
                                                                                     'multitask_regression_label_2',
                                                                                     'multitask_regression_label_3'])
        self.multitask_regression_dataset_test.__len__.return_value = 10

    def tearDown(self) -> None:
        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')

    @abstractmethod
    def test_fit_predict_evaluate(self):
        raise NotImplementedError
