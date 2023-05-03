import os
from unittest import TestCase, skip
from unittest.mock import MagicMock

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from deepmol.datasets import Dataset
from deepmol.feature_importance.shap_values import ShapValues
from deepmol.models import Model


class TestShap(TestCase):

    def setUp(self) -> None:
        self.dataset = MagicMock(spec=Dataset,
                                 ids=np.array(['1', '2', '3', '4', '5']),
                                 X=np.array([[1, 2, 3, 1], [4, 5, 6, 4], [7, 8, 9, 7], [1, 2, 3, 1], [4, 5, 6, 4]]),
                                 y=np.array([1, 0, 1, 1, 0]),
                                 n_tasks=1,
                                 feature_names=np.array(['f1', 'f2', 'f3', 'f4']),
                                 label_names=np.array(['label']),
                                 mode='classification')

        def side_effect_predict(arg):
            # randomly pick 0 or 1 len(arg) times
            return np.random.choice([0, 1], len(arg))
        rf = MagicMock(spec=RandomForestClassifier)
        rf.predict.side_effect = side_effect_predict
        self.model = MagicMock(spec=Model, model=rf)
        self.path = 'fig.png'
        self.html_path = 'fig.html'

        self.shap = ShapValues('exact', None)
        self.shap.fit(self.dataset, self.model)

    def tearDown(self) -> None:
        paths_to_remove = ['deepmol.log', self.path, self.html_path]
        # Remove each path if it exists
        for path in paths_to_remove:
            if os.path.exists(path):
                os.remove(path)

    def test_shap(self):
        self.assertIsNotNone(self.shap.shap_values)

    def test_beeswarm_plot(self):
        self.shap.beeswarm_plot(path=self.path, max_display=4)
        self.assertTrue(os.path.exists(self.path))

    def test_bar_plot(self):
        self.shap.bar_plot(path=self.path)
        self.assertTrue(os.path.exists(self.path))

    def test_sample_explanation_plot(self):
        self.shap.sample_explanation_plot(index=0, path=self.path)
        self.assertTrue(os.path.exists(self.path))

        self.shap.sample_explanation_plot(index=0, plot_type='force', path=self.path)
        self.assertTrue(os.path.exists(self.path))

    def test_feature_explanation_plot(self):
        self.shap.feature_explanation_plot(0, path=self.path)
        self.assertTrue(os.path.exists(self.path))

    def test_heatmap_plot(self):
        self.shap.heatmap_plot(path=self.path)
        self.assertTrue(os.path.exists(self.path))

    def test_positive_class_plot(self):
        self.shap.positive_class_plot(path=self.path)
        self.assertTrue(os.path.exists(self.path))

    def test_negative_class_plot(self):
        self.shap.negative_class_plot(path=self.path)
        self.assertTrue(os.path.exists(self.path))

    def test_decision_plot(self):
        with self.assertRaises(ValueError):
            self.shap.decision_plot(path=self.path)


