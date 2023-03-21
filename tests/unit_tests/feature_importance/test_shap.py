from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from deepmol.datasets import Dataset
from deepmol.feature_importance import ShapValues
from deepmol.models import Model


class TestShap(TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_shap(self):
        dataset = MagicMock(spec=Dataset,
                            ids=np.array([1, 2, 3]),
                            y=np.array([1, 2, 3]),
                            n_tasks=1,
                            label_names=np.array(['label']))

        model = MagicMock(spec=Model)
        model.predict.return_value = [1, 2, 3]

        shap = ShapValues('permutation', 'partition')
        shap.compute_shap(dataset, model)
