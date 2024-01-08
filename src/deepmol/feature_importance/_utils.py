from typing import Union

from shap import Explainer
from shap.explainers import Exact
from shap.explainers.other import Random
from shap.maskers import Independent, Masker
from shap.maskers import Partition

from shap import TreeExplainer, DeepExplainer, LinearExplainer, KernelExplainer, GPUTreeExplainer, \
    PermutationExplainer, PartitionExplainer, SamplingExplainer, AdditiveExplainer

from deepmol.models import Model

explainers = {'explainer': Explainer,
              'kernel': KernelExplainer,
              'sampling': SamplingExplainer,
              'tree': TreeExplainer,
              'gpu_tree': GPUTreeExplainer,
              'deep': DeepExplainer,
              'linear': LinearExplainer,
              'partition': PartitionExplainer,
              'permutation': PermutationExplainer,
              'additive': AdditiveExplainer,
              'exact': Exact,
              'random': Random,
              }

maskers = {'independent': Independent,
           'partition': Partition,
           }


def str_to_explainer(explainer: str) -> Explainer:
    """
    Convert a string to a SHAP explainer.

    Parameters
    ----------
    explainer: str
        The name of the explainer

    Returns
    -------
    explainer: Explainer
        The SHAP explainer.
    """
    if explainer not in explainers.keys():
        raise ValueError(f'Explainer {explainer} not supported. '
                         f'Available explainers: {list(explainers.keys())}')
    return explainers[explainer]


def str_to_masker(masker: str) -> Masker:
    """
    Convert a string to a SHAP masker.

    Parameters
    ----------
    masker: str
        The name of the masker

    Returns
    -------
    masker: Masker
        The SHAP masker.
    """
    if masker not in maskers.keys():
        raise ValueError(f'Masker {masker} not supported. '
                         f'Available maskers: {list(maskers.keys())}')
    return maskers[masker]


def get_model_instance_from_explainer(explainer: str, model: Model) -> Union[Model, callable]:
    """
    Get the model instance/prediction function based on the explainer type.

    Parameters
    ----------
    explainer: str
        The name of the explainer
    model: Model
        The model to explain

    Returns
    -------
    model_instance: Model or callable
        The model instance or prediction function.
    """
    if explainer in ['tree', 'linear']:
        return model.model
    if explainer == 'gpu_tree':
        raise ValueError('Tree explainer not supported yet! Referring to '
                         'https://github.com/slundberg/shap/issues/1136 and '
                         'https://github.com/slundberg/shap/issues/1650')
    elif explainer == 'deep':
        return model.model.model(**model.builder_kwargs)
    else:
        return model.model.predict
