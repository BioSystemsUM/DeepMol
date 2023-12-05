from copy import deepcopy

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from deepmol.models import SklearnModel
from deepmol.pipeline import Pipeline
from deepmol.pipeline.ensemble import VotingPipeline
from integration_tests.pipeline.test_pipeline import TestPipeline


class TestEnsemblePipeline(TestPipeline):

    def test_ensemble_pipeline_classification(self):
        rf = RandomForestClassifier()
        svc = SVC(probability=True)
        model_rf = SklearnModel(model=rf, model_dir='model_rf')
        model_svc = SklearnModel(model=svc, model_dir='model_svc')
        knn = KNeighborsClassifier()
        model_knn = SklearnModel(model=knn, model_dir='model_knn')
        pipeline1 = Pipeline(steps=[('model', model_rf)], path='test_pipeline_rf/')
        pipeline2 = Pipeline(steps=[('model', model_svc)], path='test_pipeline_svc/')
        pipeline3 = Pipeline(steps=[('model', model_knn)], path='test_pipeline_knn/')
        vp = VotingPipeline(pipelines=[pipeline1, pipeline2, pipeline3], voting='soft')
        vp.fit(self.dataset_descriptors)
        predictions = vp.predict(self.dataset_descriptors)
        predictions_proba = vp.predict_proba(self.dataset_descriptors)
        # TODO: add some assertions

        vp = VotingPipeline(pipelines=[pipeline1, pipeline2, pipeline3], voting='hard')
        vp.fit(self.dataset_descriptors)
        predictions = vp.predict(self.dataset_descriptors)
        predictions_proba = vp.predict_proba(self.dataset_descriptors)
        # TODO: add some assertions

        vp_w_weights = VotingPipeline(pipelines=[pipeline1, pipeline2, pipeline3], voting='soft', weights=[1, 2, 1])
        vp_w_weights.fit(self.dataset_descriptors)
        predictions = vp_w_weights.predict(self.dataset_descriptors)
        predictions_proba = vp_w_weights.predict_proba(self.dataset_descriptors)
        # TODO: add some assertions

        vp_w_weights = VotingPipeline(pipelines=[pipeline1, pipeline2, pipeline3], voting='hard', weights=[1, 2, 1])
        vp_w_weights.fit(self.dataset_descriptors)
        predictions = vp_w_weights.predict(self.dataset_descriptors)
        predictions_proba = vp_w_weights.predict_proba(self.dataset_descriptors)
        # TODO: add some assertions

    def test_enemble_pipeline_regression(self):
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
        print(dc.y)
        predictions = vp.predict(dc)
        print(predictions)
