from models.Models import Model
from metrics.Metrics import Metric
from Dataset.Dataset import Dataset
from typing import Dict, List, Optional, Tuple


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

  class GridHyperparamOpt(HyperparamOpt):
    """
    Provides simple grid hyperparameter search capabilities.
    This class performs a grid hyperparameter search over the specified
    hyperparameter space. This implementation is simple and simply does
    a direct iteration over all possible hyperparameters and doesn't use
    parallelization to speed up the search.
    """

    def hyperparam_search(
            self,
            params_dict: Dict,
            train_dataset: Dataset,
            valid_dataset: Dataset,
            metric: Metric,
            use_max: bool = True,
            logdir: Optional[str] = None,
            **kwargs,):
      """Perform hyperparams search according to params_dict.
      Each key to hyperparams_dict is a model_param. The values should
      be a list of potential values for that hyperparam.
      Parameters
      ----------
      params_dict: Dict
        Maps hyperparameter names (strings) to lists of possible
        parameter values.
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
        an instance of `dc.model.Model`, `best_hyperparams` is a
        dictionary of parameters, and `all_scores` is a dictionary mapping
        string representations of hyperparameter sets to validation
        scores.
      """
      hyperparams = params_dict.keys()
      hyperparam_vals = params_dict.values()
      for hyperparam_list in params_dict.values():
        assert isinstance(hyperparam_list, collections.Iterable)

      number_combinations = reduce(mul, [len(vals) for vals in hyperparam_vals])

      if use_max:
        best_validation_score = -np.inf
      else:
        best_validation_score = np.inf
      best_hyperparams = None
      best_model, best_model_dir = None, None
      all_scores = {}
      for ind, hyperparameter_tuple in enumerate(
              itertools.product(*hyperparam_vals)):
        model_params = {}
        print("Fitting model %d/%d" % (ind + 1, number_combinations))
        # Construction dictionary mapping hyperparameter names to values
        hyper_params = dict(zip(hyperparams, hyperparameter_tuple))
        for hyperparam, hyperparam_val in zip(hyperparams, hyperparameter_tuple):
          model_params[hyperparam] = hyperparam_val
        logger.info("hyperparameters: %s" % str(model_params))

        if logdir is not None:
          model_dir = os.path.join(logdir, str(ind))
          print("model_dir is %s" % model_dir)
          try:
            os.makedirs(model_dir)
          except OSError:
            if not os.path.isdir(model_dir):
              logger.info("Error creating model_dir, using tempfile directory")
              model_dir = tempfile.mkdtemp()
        else:
          model_dir = tempfile.mkdtemp()
        model_params['model_dir'] = model_dir
        model = self.model_builder(**model_params)
        model.fit(train_dataset)
        try:
          model.save()
        # Some models autosave
        except NotImplementedError:
          pass

        multitask_scores = model.evaluate(valid_dataset, [metric])
        valid_score = multitask_scores[metric.name]
        hp_str = _convert_hyperparam_dict_to_filename(hyper_params)
        all_scores[hp_str] = valid_score

        if (use_max and valid_score >= best_validation_score) or (
                not use_max and valid_score <= best_validation_score):
          best_validation_score = valid_score
          best_hyperparams = hyperparameter_tuple
          if best_model_dir is not None:
            shutil.rmtree(best_model_dir)
          best_model_dir = model_dir
          best_model = model
        else:
          shutil.rmtree(model_dir)

        print("Model %d/%d, Metric %s, Validation set %s: %f" %
                    (ind + 1, number_combinations, metric.name, ind, valid_score))
        print("\tbest_validation_score so far: %f" % best_validation_score)
      if best_model is None:
        print("No models trained correctly.")
        # arbitrarily return last model
        best_model, best_hyperparams = model, hyperparameter_tuple
        return best_model, best_hyperparams, all_scores
      multitask_scores = best_model.evaluate(train_dataset, [metric])
      train_score = multitask_scores[metric.name]
      print("Best hyperparameters: %s" % str(best_hyperparams))
      print("train_score: %f" % train_score)
      print("validation_score: %f" % best_validation_score)
      return best_model, best_hyperparams, all_scores