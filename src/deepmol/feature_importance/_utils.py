from shap import Explainer
from shap.explainers import Additive, Exact, GPUTree, Permutation, Partition, Tree, Linear, Sampling, Deep
from shap.maskers import Independent, Impute, Text, Masker
from shap.maskers import Partition as PartitionM

explainers = {'additive': Additive,
              'exact': Exact,
              'gpu_tree': GPUTree,  # not sure if it is an explainer
              'permutation': Permutation,
              'partition': Partition,
              'tree': Tree,
              'linear': Linear,
              'sampling': Sampling,
              'deep': Deep
              }

maskers = {'independent': Independent,  # data, max_samples=100
           'partition': PartitionM,  # data, max_samples=100, clustering="correlation"
           'impute': Impute,  # data, method="linear"
           'text': Text,  # tokenizer=None, mask_token=None, collapse_mask_token="auto", output_type="string"
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
    else:
        raise ValueError(f'Unknown masker: {masker}')
