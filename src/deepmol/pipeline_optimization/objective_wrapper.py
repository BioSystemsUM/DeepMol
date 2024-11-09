import os
import shutil
from abc import abstractmethod
from copy import copy

import optuna
from optuna import Trial

from deepmol.pipeline import Pipeline
from deepmol.utils.decorators import timeout

import traceback

import timeout_decorator


class Objective:
    """
    Wrapper for the objective function of the pipeline optimization.
    It creates, saves and evaluates pipelines for each trial.

    Parameters
    ----------
    """

    def __init__(self, objective_steps, study, direction, save_top_n, trial_timeout=86400):
        """
        Initialize the objective function.
        """
        self.objective_steps = objective_steps
        self.study = study
        self.direction = direction
        self.save_top_n = save_top_n
        self.save_dir = study.study_name
        self.trial_timeout = trial_timeout

        if os.name == "nt":
            self.run_function = self.run_with_timeout_windows()
        else:
            self.run_function = self.run_with_timeout_unix()
    
    def run_with_timeout_windows(self):

        @timeout(self.trial_timeout)
        def run_with_timeout(trial):
            return self._run(trial)
        
        return run_with_timeout
    
    def run_with_timeout_unix(self):

        @timeout_decorator.timeout(self.trial_timeout, use_signals=True)
        def run_with_timeout(trial):
            return self._run(trial)
        
        return run_with_timeout

    @abstractmethod
    def _run(self, trial):
        pass

    def __call__(self, trial: Trial):
        """
        Create and evaluate a pipeline for a given trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial object that stores the hyperparameters.
        """
        try:
            return self.run_function(trial)
        except ValueError as e:
            print(e)
            traceback.print_exc()
            return float('inf') if self.direction == 'minimize' else float('-inf')
        
        except Exception as e:
            print(e)
            traceback.print_exc()
            return float('inf') if self.direction == 'minimize' else float('-inf')


class ObjectiveTrainEval(Objective):
    """
    Wrapper for the objective function of the pipeline optimization.
    It creates and saves pipelines for each trial and evaluates them on the test dataset.

    Parameters
    ----------
    objective_steps : callable
        Function that returns the steps of the pipeline for a given trial.
    study : optuna.study.Study
        Study object that stores the optimization history.
    direction : str or optuna.study.StudyDirection
        Direction of the optimization (minimize or maximize).
    train_dataset : deepmol.datasets.Dataset
        Dataset used for training the pipeline.
    test_dataset : deepmol.datasets.Dataset
        Dataset used for evaluating the pipeline.
    metric : deepmol.metrics.Metric
        Metric used for evaluating the pipeline.
    save_top_n : int
        Number of best pipelines to save.
    **kwargs
        Additional keyword arguments passed to the objective_steps function.
    """

    def __init__(self, objective_steps, study, direction, save_top_n, trial_timeout=86400, **kwargs):
        """
        Initialize the objective function.
        """
        super().__init__(objective_steps, study, direction, save_top_n, trial_timeout=trial_timeout)
        if "train_dataset" not in kwargs or "test_dataset" not in kwargs or "metric" not in kwargs:
            raise ValueError("train_dataset, test_dataset and metric must be passed as keyword arguments.")
        self.train_dataset = kwargs.pop('train_dataset')
        self.test_dataset = kwargs.pop('test_dataset')
        self.metric = kwargs.pop('metric')
        self.kwargs = kwargs

    def _run(self, trial):
        """
        Create and evaluate a pipeline for a given trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial object that stores the hyperparameters.
        """

        train_dataset = copy(self.train_dataset)
        test_dataset = copy(self.test_dataset)
        trial_id = str(trial.number)
        path = os.path.join(self.save_dir, f'trial_{trial_id}')
        pipeline = Pipeline(steps=self.objective_steps(trial, **self.kwargs), path=path)
        if pipeline.steps[-1][1].__class__.__name__ == 'KerasModel':
            pipeline.fit(train_dataset, validation_dataset=test_dataset)
        else:
            pipeline.fit(train_dataset)
        score = pipeline.evaluate(test_dataset, [self.metric])[0][self.metric.name]
        if score is None:
            score = float('-inf') if self.direction == 'maximize' else float('inf')

        best_scores = self.study.user_attrs['best_scores']
        min_score = min(best_scores.values()) if len(best_scores) > 0 else float('inf')
        max_score = max(best_scores.values()) if len(best_scores) > 0 else float('-inf')
        update_score = (self.direction == 'maximize' and score > min_score) or (
                self.direction == 'minimize' and score < max_score)

        if len(best_scores) < self.save_top_n or update_score:
            pipeline.save()
            best_scores.update({trial_id: score})

            if len(best_scores) > self.save_top_n:
                if self.direction == 'maximize':
                    min_score_id = min(best_scores, key=best_scores.get)
                    del best_scores[min_score_id]
                    if os.path.exists(os.path.join(self.save_dir, f'trial_{min_score_id}')):
                        shutil.rmtree(os.path.join(self.save_dir, f'trial_{min_score_id}'))
                else:
                    max_score_id = max(best_scores, key=best_scores.get)
                    del best_scores[max_score_id]
                    if os.path.exists(os.path.join(self.save_dir, f'trial_{max_score_id}')):
                        shutil.rmtree(os.path.join(self.save_dir, f'trial_{max_score_id}'))

        self.study.set_user_attr('best_scores', best_scores)
        return score

        
