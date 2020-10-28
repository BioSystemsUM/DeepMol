from models.Models import Model
from metrics.Metrics import Metric
from Dataset.Dataset import Dataset



class HyperparamOpt(object):
  """Abstract superclass for hyperparameter search classes.
  """

  def __init__(self, model_builder: Model):
    """Initialize Hyperparameter Optimizer.
    Note this is an abstract constructor which should only be used by
    subclasses.
    Parameters
    ----------
    model_builder: constructor function.
      This parameter must be constructor function which returns an
      object which is an instance of `dc.models.Model`. This function
      must accept two arguments, `model_params` of type `dict` and
      `model_dir`, a string specifying a path to a model directory.
      See the example.
    """
    if self.__class__.__name__ == "HyperparamOpt":
      raise ValueError(
          "HyperparamOpt is an abstract superclass and cannot be directly instantiated. \
          You probably want to instantiate a concrete subclass instead.")
    self.model_builder = model_builder

  def hyperparam_search(
      self,
      params_dict: Dict[str, Any],
      train_dataset: Dataset,
      valid_dataset: Dataset,
      metric: Metric,
      use_max: bool = True,
      logdir: Optional[str] = None,
      **kwargs) -> Tuple[Model, Dict[str, Any], Dict[str, float]]:
    """Conduct Hyperparameter search.
    This method defines the common API shared by all hyperparameter
    optimization subclasses. Different classes will implement
    different search methods but they must all follow this common API.
    Parameters
    ----------
    params_dict: Dict
      Dictionary mapping strings to values. Note that the
      precise semantics of `params_dict` will change depending on the
      optimizer that you're using. Depending on the type of
      hyperparameter optimization, these values can be
      ints/floats/strings/lists/etc. Read the documentation for the
      concrete hyperparameter optimization subclass you're using to
      learn more about what's expected.
    train_dataset: Dataset
      dataset used for training
    valid_dataset: Dataset
      dataset used for validation(optimization on valid scores)
    metric: Metric
      metric used for evaluation
    use_max: bool, optional
      If True, return the model with the highest score. Else return
      model with the minimum score.
    logdir: str, optional
      The directory in which to store created models. If not set, will
      use a temporary directory.
    Returns
    -------
    Tuple[`best_model`, `best_hyperparams`, `all_scores`]
      `(best_model, best_hyperparams, all_scores)` where `best_model` is
      an instance of `dc.models.Model`, `best_hyperparams` is a
      dictionary of parameters, and `all_scores` is a dictionary mapping
      string representations of hyperparameter sets to validation
      scores.
    """
    raise NotImplementedError