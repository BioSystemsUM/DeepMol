from shap import Explainer
from shap.explainers import Additive, Exact, GPUTree, Permutation, Partition, Tree, Linear, Sampling, Deep
from shap.explainers.other import Random
from shap.maskers import Independent, Masker
from shap.maskers import Partition as PartitionM

explainers = {'explainer': Explainer,
              'tree': Tree,
              'gpu_tree': GPUTree,
              'linear': Linear,
              'permutation': Permutation,
              'partition': Partition,
              'sampling': Sampling,
              'additive': Additive,
              'deep': Deep,
              'exact': Exact,
              'random': Random
              }

maskers = {'independent': Independent,
           'partition': PartitionM,
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


def masker_args(masker: str, **kwargs) -> dict:
    """
    Get the arguments of a SHAP masker.

    Parameters
    ----------
    masker: str
        The name of the masker
    **kwargs: dict
        Additional arguments for the masker.

    Returns
    -------
    args: dict
        The arguments of the masker.
    """
    if masker == 'independent':
        return {'max_samples': kwargs.get('max_samples', 100)}
    elif masker == 'partition':
        return {'max_samples': kwargs.get('max_samples', 100),
                'clustering': kwargs.get('clustering', 'correlation')}
    elif masker == 'impute':
        return {'method': kwargs.get('method', 'linear')}
    else:
        raise ValueError(f'Unknown masker: {masker}')
