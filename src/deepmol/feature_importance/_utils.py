from shap import Explainer
from shap.explainers import Additive, Exact, GPUTree, Permutation, Partition, Tree, Linear, Sampling, Deep
from shap.explainers._gradient import Gradient
from shap.explainers._kernel import Kernel
from shap.explainers.other import Coefficent, Random, LimeTabular, Maple, TreeMaple, TreeGain
from shap.maskers import Independent, Impute, Text, Masker, Fixed, Image
from shap.maskers import Partition as PartitionM

explainers = {'explainer': Explainer,
              'tree': Tree,
              'gpu_tree': GPUTree,
              'linear': Linear,
              'permutation': Permutation,
              'partition': Partition,
              'sampling': Sampling,
              'additive': Additive,
              'gradient': Gradient,
              'deep': Deep,
              'exact': Exact,
              'kernel': Kernel,
              'coefficient': Coefficent,
              'random': Random,
              'lime_tabular': LimeTabular,
              'maple': Maple,
              'tree_maple': TreeMaple,
              'tree_gain': TreeGain,
              }

maskers = {'independent': Independent,
           'partition': PartitionM,
           'impute': Impute,
           'fixed': Fixed,
           'text': Text,
           'image': Image,
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
    elif masker == 'text':
        return {'tokenizer': kwargs.get('tokenizer', None),
                'mask_token': kwargs.get('mask_token', None),
                'collapse_mask_token': kwargs.get('collapse_mask_token', 'auto'),
                'output_type': kwargs.get('output_type', 'string')}
    elif masker == 'fixed':
        return {}
    elif masker == 'image':
        return {'mask_value': kwargs.get('mask_value'),
                'shape': kwargs.get('shape', None)}
    else:
        raise ValueError(f'Unknown masker: {masker}')
