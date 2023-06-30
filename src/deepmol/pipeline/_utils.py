from deepmol.base import Predictor
from deepmol.models import SklearnModel, KerasModel, DeepChemModel

MODEL_TYPES = {'sklearn': SklearnModel,
               'keras': KerasModel,
               'deepchem': DeepChemModel}


def _get_predictor_instance(model_type: str) -> Predictor:
    """
    Returns an instance of the predictor corresponding to the model type.

    Parameters
    ----------
    model_type: str
        Type of model.

    Returns
    -------
    Predictor
        Instance of the predictor corresponding to the model type.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Model type {model_type} not supported.")
    return MODEL_TYPES[model_type]
