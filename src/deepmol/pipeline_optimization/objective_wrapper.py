import os
import shutil

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
        self.save_dir = os.path.join(os.getcwd(), study.study_name)
        self.kwargs = kwargs

    def __call__(self, trial: Trial):
        """
        Create and evaluate a pipeline for a given trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial object that stores the hyperparameters.
        """
        trial_id = str(trial.number)
        path = os.path.join(self.save_dir, f'trial_{trial_id}')
        pipeline = Pipeline(steps=self.objective_steps(trial, **self.kwargs), path=path)
        pipeline.fit(self.train_dataset)
        score = pipeline.evaluate(self.test_dataset, [self.metric])[0][self.metric.name]
        # save pipeline if score is in top n scores
        if len(self.study.user_attrs['best_scores']) < self.save_top_n:
            pipeline.save()
            self.study.set_user_attr('best_scores', {**self.study.user_attrs['best_scores'], trial_id: score})
        else:
            if self.direction == 'maximize':
                min_score = min(self.study.user_attrs['best_scores'].values())
                min_score_id = [k for k, v in self.study.user_attrs['best_scores'].items() if v == min_score][0]
                if score > min_score:
                    pipeline.save()
                    self.study.set_user_attr('best_scores', {**self.study.user_attrs['best_scores'], trial_id: score})
                    new_scores = self.study.user_attrs['best_scores']
                    del new_scores[min_score_id]
                    self.study.set_user_attr('best_scores', {**new_scores})
                    shutil.rmtree(os.path.join(self.save_dir, f'trial_{min_score_id}'))
            else:  # minimize
                max_score = max(self.study.user_attrs['best_scores'].values())
                max_score_id = [k for k, v in self.study.user_attrs['best_scores'].items() if v == max_score][0]
                if score < max_score:
                    pipeline.save()
                    self.study.set_user_attr('best_scores',
                                             {**self.study.user_attrs['best_scores'], trial_id: score})
                    new_scores = self.study.user_attrs['best_scores']
                    del new_scores[max_score_id]
                    self.study.set_user_attr('best_scores', {**new_scores})
                    shutil.rmtree(os.path.join(self.save_dir, f'trial_{max_score_id}'))
        return score
