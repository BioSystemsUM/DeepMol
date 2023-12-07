import os
import shutil
from copy import deepcopy

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from deepmol.metrics import Metric
from deepmol.models import SklearnModel
from deepmol.pipeline import Pipeline
from deepmol.pipeline.ensemble import VotingPipeline
from integration_tests.pipeline.test_pipeline import TestPipeline


class TestEnsemblePipeline(TestPipeline):

    def tearDown(self) -> None:
        for f in os.listdir():
            if f.startswith('deepmol.log'):
                os.remove(f)
            if f.startswith('model_') or f.startswith('test_pipeline_') or f.startswith('test_ensemble_pipeline_'):
                if os.path.isdir(f):
                    shutil.rmtree(f)

    def test_ensemble_pipeline_classification(self):
        rf = RandomForestClassifier()
        svc = SVC(probability=True)
        model_rf = SklearnModel(model=rf, model_dir='model_rf')
        rf.fit(self.dataset_descriptors.X, self.dataset_descriptors.y)
        model_svc = SklearnModel(model=svc, model_dir='model_svc')
        knn = KNeighborsClassifier()
        model_knn = SklearnModel(model=knn, model_dir='model_knn')
        pipeline1 = Pipeline(steps=[('model', model_rf)], path='test_pipeline_rf/')
        pipeline2 = Pipeline(steps=[('model', model_svc)], path='test_pipeline_svc/')
        pipeline3 = Pipeline(steps=[('model', model_knn)], path='test_pipeline_knn/')
        vp = VotingPipeline(pipelines=[pipeline1, pipeline2, pipeline3], voting='soft')
        vp.fit(self.dataset_descriptors)
        test_dataset = deepcopy(self.dataset_descriptors).select_to_split(np.arange(0, 15))
        predictions = vp.predict(test_dataset)
        self.assertEqual(len(predictions), len(test_dataset.y))
        score = vp.evaluate(test_dataset, metrics=[Metric(accuracy_score)])
        self.assertIsInstance(score[0]['accuracy_score'], float)

        vp = VotingPipeline(pipelines=[pipeline1, pipeline2, pipeline3], voting='hard')
        vp.fit(self.dataset_descriptors)
        predictions = vp.predict(test_dataset)
        self.assertEqual(len(predictions), len(test_dataset.y))
        score = vp.evaluate(test_dataset, metrics=[Metric(accuracy_score)])
        self.assertIsInstance(score[0]['accuracy_score'], float)

        vp_w_weights = VotingPipeline(pipelines=[pipeline1, pipeline2, pipeline3], voting='soft', weights=[1, 2, 1])
        vp_w_weights.fit(self.dataset_descriptors)
        predictions = vp_w_weights.predict(test_dataset)
        self.assertEqual(len(predictions), len(test_dataset.y))
        score = vp_w_weights.evaluate(test_dataset, metrics=[Metric(accuracy_score)])
        self.assertIsInstance(score[0]['accuracy_score'], float)

        vp_w_weights = VotingPipeline(pipelines=[pipeline1, pipeline2, pipeline3], voting='hard',
                                      weights=[0.45, 0.25, 0.3])
        vp_w_weights.fit(self.dataset_descriptors)
        predictions = vp_w_weights.predict(test_dataset)
        self.assertEqual(len(predictions), len(test_dataset.y))
        self.assertEqual(len(predictions), len(test_dataset.y))
        score = vp_w_weights.evaluate(test_dataset, metrics=[Metric(accuracy_score)])
        self.assertIsInstance(score[0]['accuracy_score'], float)

        vp_w_weights.save('test_ensemble_pipeline_classification')
        vp_w_weights_loaded = VotingPipeline.load('test_ensemble_pipeline_classification')
        predictions_loaded = vp_w_weights_loaded.predict(test_dataset)
        self.assertEqual(len(predictions_loaded), len(test_dataset.y))
        self.assertTrue(np.array_equal(predictions_loaded, predictions))
        score_loaded = vp_w_weights.evaluate(test_dataset, metrics=[Metric(accuracy_score)])
        self.assertIsInstance(score_loaded[0]['accuracy_score'], float)
        self.assertEqual(score_loaded[0]['accuracy_score'], score[0]['accuracy_score'])

    def test_ensemble_pipeline_regression(self):
        rf = RandomForestRegressor()
        svc = SVR()
        knn = KNeighborsRegressor()
        model_rf = SklearnModel(model=rf, model_dir='model_rf')
        model_svc = SklearnModel(model=svc, model_dir='model_svc')
        model_knn = SklearnModel(model=knn, model_dir='model_knn')
        pipeline1 = Pipeline(steps=[('model', model_rf)], path='test_pipeline_rf/')
        pipeline2 = Pipeline(steps=[('model', model_svc)], path='test_pipeline_svc/')
        pipeline3 = Pipeline(steps=[('model', model_knn)], path='test_pipeline_knn/')
        vp = VotingPipeline(pipelines=[pipeline1, pipeline2, pipeline3], weights=[1, 2, 1])
        dc = deepcopy(self.dataset_descriptors)
        dc.mode = 'regression'
        # float values between 0 and 10
        dc._y = np.random.rand(len(dc.y)) * 10
        vp.fit(dc)
        predictions = vp.predict(dc)
        self.assertEqual(len(predictions), len(dc.y))
