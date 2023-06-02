from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.pipeline import Pipeline
from deepmol.pipeline_optimization._deepchem_models_objectives import gat_model_steps, gcn_model_steps, \
    pagtn_model_steps, attentive_fp_model_steps, mpnn_model_steps, megnet_model_steps, cnn_model_steps, \
    multitask_classifier_model_steps, multitask_irv_classifier_model_steps, \
    progressive_multitask_classifier_model_steps, robust_multitask_classifier_model_steps, sc_score_model_steps, \
    atomic_conv_model_steps
from deepmol.pipeline_optimization._featurizer_objectives import _get_featurizer
from deepmol.pipeline_optimization._standardizer_objectives import _get_standardizer

# TODO: add more models
_BASE_MODELS = []
_RNN_MODELS = []
_CNN_MODELS = []


# TODO: How to deal with incompatible steps? (e.g. some deepchem featurizers only work with some deepchem models)


def _get_preset(preset: str) -> callable:
    return singletask_classification_objective


def singletask_classification_objective(trial,
                                        train_dataset: Dataset,
                                        test_dataset: Dataset,
                                        metric: Metric) -> float:
    # TODO: "mpnn_model" is not working (DeepChem-> AttributeError: 'GraphData' object has no attribute 'get_num_atoms')
    # TODO: "megnet_model" is not working (error with torch_geometric (extra_requirement))
    # TODO: "cnn_model" is not working (raise PicklingError(
    #  pickle.PicklingError: Can't pickle <class '_thread.lock'>: it's not found as _thread.lock))
    # TODO: "multitask_irv_classifier_model" not working (needs 1D featurizer + irv transformer but it is not working)
    # TODO: "progressive_multitask_classifier_model" not working (ValueError: Index out of range using input dim 1;
    #  input has only 1 dims for '{{node strided_slice_1}} ...)
    # TODO: "sc_score_model" not working (Input 0 of layer "dense" is incompatible with the layer: expected axis -1 of
    #  input shape to have value 2048, but received input with shape (100, 1))
    # steps = trial.suggest_categorical("steps", ["gat_model", "gcn_model", "attentive_fp_model", "pagtn_model",
    #                                             "multitask_classifier_model",
    #                                             "robust_multitask_classifier_model", "atomic_conv_model"])
    steps = trial.suggest_categorical("steps", ["atomic_conv_model"])
    if steps == "gat_model":
        gat_kwargs = {'n_tasks': 1, 'mode': 'classification'}
        steps = gat_model_steps(trial=trial, gat_kwargs=gat_kwargs)
    elif steps == "gcn_model":
        gcn_kwargs = {'n_tasks': 1, 'mode': 'classification'}
        steps = gcn_model_steps(trial=trial, gcn_kwargs=gcn_kwargs)
    elif steps == "attentive_fp_model":
        attentive_fp_kwargs = {'n_tasks': 1, 'mode': 'classification'}
        steps = attentive_fp_model_steps(trial=trial, attentive_fp_kwargs=attentive_fp_kwargs)
    elif steps == "pagtn_model":
        patgn_kwargs = {'n_tasks': 1, 'mode': 'classification'}
        steps = pagtn_model_steps(trial=trial, patgn_kwargs=patgn_kwargs)
    # elif steps == "mpnn_model":
    #     mpnn_kwargs = {'n_tasks': 1, 'mode': 'classification'}
    #     steps = mpnn_model_steps(trial=trial, mpnn_kwargs=mpnn_kwargs)
    # elif steps == "megnet_model":
    #     megnet_kwargs = {'n_tasks': 1, 'mode': 'classification'}
    #     steps = megnet_model_steps(trial=trial, megnet_kwargs=megnet_kwargs)
    # elif steps == "cnn_model":
    #     n_features = 1024
    #     dims = 1
    #     featurizer = MorganFingerprint(size=1024)
    #     cnn_kwargs = {'n_tasks': 1, 'mode': 'classification', 'n_features': n_features, 'dims': dims}
    #     steps = cnn_model_steps(trial=trial, n_features=n_features, dims=dims, cnn_kwargs=cnn_kwargs)
    #     steps = (('featurizer', featurizer), steps[0])
    elif steps == "multitask_classifier_model":
        featurizer = _get_featurizer(trial, '1D')
        n_features = len(featurizer.feature_names)
        multitask_classifier_kwargs = {'n_tasks': 1, 'n_features': n_features}
        model_step = multitask_classifier_model_steps(trial=trial,
                                                      multitask_classifier_kwargs=multitask_classifier_kwargs)
        steps = [('featurizer', featurizer), model_step]
    # elif steps == "multitask_irv_classifier_model":
    #     featurizer = _get_featurizer(trial, '1D')
    #     multitask_irv_classifier_kwargs = {'n_tasks': 1}
    #     model_step = multitask_irv_classifier_model_steps(trial=trial,
    #                                                       multitask_irv_classifier_kwargs=multitask_irv_classifier_kwargs)
    #     steps = [('featurizer', featurizer), model_step]
    # elif steps == "progressive_multitask_classifier_model":
    #     featurizer = _get_featurizer(trial, '1D')
    #     n_features = len(featurizer.feature_names)
    #     progressive_multitask_classifier_kwargs = {'n_tasks': 1, 'n_features': n_features}
    #     model_step = progressive_multitask_classifier_model_steps(trial=trial,
    #                                                               progressive_multitask_classifier_kwargs=progressive_multitask_classifier_kwargs)
    #     steps = [('featurizer', featurizer), model_step]
    elif steps == "robust_multitask_classifier_model":
        featurizer = _get_featurizer(trial, '1D')
        n_features = len(featurizer.feature_names)
        robust_multitask_classifier_kwargs = {'n_tasks': 1, 'n_features': n_features}
        model_step = robust_multitask_classifier_model_steps(trial=trial,
                                                             robust_multitask_classifier_kwargs=robust_multitask_classifier_kwargs)
        steps = [('featurizer', featurizer), model_step]
    # elif steps == "sc_score_model":
    #     featurizer = _get_featurizer(trial, '1D')
    #     n_features = len(featurizer.feature_names)
    #     sc_score_kwargs = {'n_features': n_features}
    #     model_step = sc_score_model_steps(trial=trial, sc_score_kwargs=sc_score_kwargs)
    #     steps = [('featurizer', featurizer), model_step]
    elif steps == "atomic_conv_model":
        atomic_conv_kwargs = {'n_tasks': 1}
        steps = atomic_conv_model_steps(trial=trial, atomic_conv_kwargs=atomic_conv_kwargs)
    print(trial.params)
    pipeline = Pipeline(steps=steps)
    pipeline.fit_transform(train_dataset)
    score = pipeline.evaluate(test_dataset, [metric])[0][metric.name]
    return score


def heavy_objective(trial, train_dataset, test_dataset, metric) -> callable:
    # model
    model = trial.suggest_categorical("model", _BASE_MODELS)
    # standardizer
    standardizer = _get_standardizer(trial)
    # featurizer
    featurizer = _get_featurizer(trial, '1D')
    # scaler
    scaler = trial.suggest_categorical("scaler", _SCALERS)
    # feature_selector
    feature_selector = trial.suggest_categorical("feature_selector", _FEATURE_SELECTORS)
    steps = [('standardizer', standardizer), ('featurizer', featurizer), ('scaler', scaler),
             ('feature_selector', feature_selector), ('model', model)]
    pipeline = Pipeline(steps=steps)
    pipeline.fit_transform(train_dataset)
    score = pipeline.evaluate(test_dataset, metric)[0][metric.name]
    return score
