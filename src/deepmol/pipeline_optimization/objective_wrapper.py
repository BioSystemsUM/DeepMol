import os
import shutil
import warnings
from copy import copy

from optuna import Trial

from deepmol.pipeline import Pipeline


class Objective:
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

    def __init__(self, objective_steps, study, direction, train_dataset, test_dataset, metric, save_top_n, **kwargs):
        """
        Initialize the objective function.
        """
        self.objective_steps = objective_steps
        self.study = study
        self.direction = direction
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.metric = metric
        self.save_top_n = save_top_n
        self.save_dir = study.study_name
        self.kwargs = kwargs

    def __call__(self, trial: Trial):
        """
        Create and evaluate a pipeline for a given trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial object that stores the hyperparameters.
        """
        try:
            train_dataset = copy(self.train_dataset)
            test_dataset = copy(self.test_dataset)
            trial_id = str(trial.number)
            path = os.path.join(self.save_dir, f'trial_{trial_id}')
            pipeline = Pipeline(steps=self.objective_steps(trial, **self.kwargs), path=path)
            pipeline.fit(train_dataset)
            score = pipeline.evaluate(test_dataset, [self.metric])[0][self.metric.name]

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
                        shutil.rmtree(os.path.join(self.save_dir, f'trial_{min_score_id}'))
                    else:
                        max_score_id = max(best_scores, key=best_scores.get)
                        del best_scores[max_score_id]
                        shutil.rmtree(os.path.join(self.save_dir, f'trial_{max_score_id}'))

            self.study.set_user_attr('best_scores', best_scores)
            return score
        except ValueError as e:
            print(e)
            return float('inf') if self.direction == 'minimize' else float('-inf')
        except Exception as e:
            print(e)
            return float('inf') if self.direction == 'minimize' else float('-inf')
