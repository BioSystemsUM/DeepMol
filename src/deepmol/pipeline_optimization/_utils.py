from copy import deepcopy, copy

from deepchem.models import TextCNNModel

from deepmol.base import DatasetTransformer
from deepmol.compound_featurization import SmilesSeqFeat
from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.pipeline import Pipeline
from deepmol.pipeline_optimization._deepchem_models_objectives import gat_model_steps, gcn_model_steps, \
    pagtn_model_steps, attentive_fp_model_steps, mpnn_model_steps, megnet_model_steps, cnn_model_steps, \
    multitask_classifier_model_steps, multitask_irv_classifier_model_steps, \
    progressive_multitask_classifier_model_steps, robust_multitask_classifier_model_steps, sc_score_model_steps, \
    chem_ception_model_steps, dag_model_steps, graph_conv_model_steps, smiles_to_vec_model_steps, text_cnn_model_steps, \
    weave_model_steps
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
    steps = trial.suggest_categorical("steps", ["gat_model", "gcn_model", "attentive_fp_model", "pagtn_model",
                                                "multitask_classifier_model",
                                                "robust_multitask_classifier_model", "chem_ception_model", "dag_model"
                                                "graph_conv_model", "smiles_to_vec_model", "text_cnn_model",
                                                "weave_model"])
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
    elif steps == "chem_ception_model":
        chem_ception_kwargs = {'n_tasks': 1}
        steps = chem_ception_model_steps(trial=trial, chem_ception_kwargs=chem_ception_kwargs)
    elif steps == "dag_model":
        dag_kwargs = {'n_tasks': 1, 'mode': 'classification'}
        steps = dag_model_steps(trial=trial, dag_kwargs=dag_kwargs)
    elif steps == "graph_conv_model":
        graph_conv_kwargs = {'n_tasks': 1, 'mode': 'classification'}
        steps = graph_conv_model_steps(trial=trial, graph_conv_kwargs=graph_conv_kwargs)
    elif steps == "smiles_to_vec_model":
        smiles_to_vec_kwargs = {'n_tasks': 1, 'mode': 'classification'}
        dataset_copy = deepcopy(train_dataset)
        ssf = SmilesSeqFeat()
        ssf.fit_transform(dataset_copy)
        chat_to_idx = ssf.char_to_idx
        smiles_to_vec_kwargs['char_to_idx'] = chat_to_idx
        steps = smiles_to_vec_model_steps(trial=trial, smiles_to_vec_kwargs=smiles_to_vec_kwargs)
    elif steps == "text_cnn_model":
        text_cnn_kwargs = {'n_tasks': 1, 'mode': 'classification'}
        max_length = max([len(smile) for smile in train_dataset.smiles])

        def prepare_dataset_for_textcnn(dataset, pad_char='E'):
            padded_smiles = [smile.ljust(max_length, pad_char) for smile in dataset.smiles]
            truncated_smiles = [smile[:max_length] for smile in padded_smiles]
            dataset._X = truncated_smiles
            dataset._ids = truncated_smiles
            return dataset
        padded_train_smiles = prepare_dataset_for_textcnn(train_dataset).ids
        fake_dataset = copy(train_dataset)
        fake_dataset._ids = padded_train_smiles
        char_dict, seq_length = TextCNNModel.build_char_dict(fake_dataset)
        text_cnn_kwargs['char_dict'] = char_dict
        print(char_dict)
        text_cnn_kwargs['seq_length'] = seq_length
        padder = DatasetTransformer(prepare_dataset_for_textcnn)
        steps = [('padder', padder), text_cnn_model_steps(trial=trial, text_cnn_kwargs=text_cnn_kwargs)]
    elif steps == "weave_model":
        weave_kwargs = {'n_tasks': 1, 'mode': 'classification'}
        steps = weave_model_steps(trial=trial, weave_kwargs=weave_kwargs)
    else:
        raise ValueError("Unknown model: %s" % steps)
    print(trial.params)
    pipeline = Pipeline(steps=steps)
    pipeline.fit_transform(train_dataset)
    score = pipeline.evaluate(test_dataset, [metric])[0][metric.name]
    # TODO: save top pipelines (arg in main class)
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
