from pathlib import Path

import pandas as pd
import shap
from matplotlib import pyplot as plt
from shap import Explanation

from deepmol.datasets import Dataset
from deepmol.feature_importance._utils import str_to_explainer, str_to_masker, get_model_instance_from_explainer
from deepmol.loggers import Logger
from deepmol.models.models import Model


class ShapValues:
    """
    SHAP (SHapley Additive exPlanations) wrapper for DeepMol
    It allows to compute and analyze the SHAP values of DeepMol models.
    """

    def __init__(self, explainer: str = 'permutation', masker: str = None) -> None:
        """
        Initialize the ShapValues object

        Parameters
        ----------
        explainer: str
            The explainer to use. It can be one of the following:
            - 'explainer': Explainer
            (https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer)
            - 'permutation': Permutation explainer
            (https://shap.readthedocs.io/en/latest/generated/shap.explainers.Permutation.html#shap.explainers.Permutation)
            - 'exact': Exact explainer
            (https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/explainers/Exact.html?highlight=exact#Exact-explainer)
            - 'additive': Additive explainer
            (https://shap.readthedocs.io/en/latest/generated/shap.explainers.Additive.html)
            - 'tree': Tree explainer
            (https://shap.readthedocs.io/en/latest/generated/shap.explainers.Tree.html)
            - 'gpu_tree': GPU Tree explainer
            (https://shap.readthedocs.io/en/latest/generated/shap.explainers.GPUTree.html)
            - 'partition': Partition explainer
            (https://shap.readthedocs.io/en/latest/generated/shap.explainers.Partition.html)
            - 'linear': Linear explainer
            (https://shap.readthedocs.io/en/latest/generated/shap.explainers.Linear.html)
            - 'sampling': Sampling explainer
            (https://shap.readthedocs.io/en/latest/generated/shap.explainers.Sampling.html)
            - 'deep': Deep explainer
            (https://github.com/slundberg/shap/blob/master/shap/explainers/_deep/deep_tf.py) and
            (https://github.com/slundberg/shap/blob/master/shap/explainers/_deep/deep_pytorch.py)
            - 'kernel': Kernel explainer
            (https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py)
            - 'random': Random explainer
            (https://shap.readthedocs.io/en/latest/generated/shap.explainers.other.Random.html)

        masker: str
            The masker to use. It can be one of the following:
            - 'independent': Independent masker
            (https://shap.readthedocs.io/en/latest/generated/shap.maskers.Independent.html)
            - 'partition': Partition masker
            (https://shap.readthedocs.io/en/latest/generated/shap.maskers.Partition.html)
        """
        self.explainer = str_to_explainer(explainer)
        self.explainer_str = explainer
        self.masker = str_to_masker(masker) if masker is not None else None
        self.shap_values = None
        self.mode = None
        self.logger = Logger()

    def fit(self, dataset: Dataset, model: Model, **kwargs) -> Explanation:
        """
        Compute the SHAP values for the given dataset and model.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the SHAP values for.
        model: Model
            The model to compute the SHAP values for.
        kwargs: dict
            Additional arguments for the SHAP explainer.

        Returns
        -------
        shap_values: np.array
            The SHAP values.
        """
        self.mode = dataset.mode
        if self.explainer_str == 'deep':
            data = dataset.X
            if self.masker is not None:
                # TODO: check this
                raise ValueError('DeepExplainer does not support maskers.')
        else:
            data = pd.DataFrame(dataset.X, columns=dataset.feature_names, dtype=float)
        model_instance = get_model_instance_from_explainer(self.explainer_str, model)
        if self.masker is not None:
            masker = self.masker(data)
            if self.explainer_str in ['sampling', 'kernel']:
                explainer = self.explainer(model_instance, data=data, masker=masker, **kwargs)
            else:
                explainer = self.explainer(model_instance, masker=masker, **kwargs)
        else:
            explainer = self.explainer(model_instance, data, **kwargs)
        try:
            shap_values = explainer(data)
        except Exception as e:
            self.logger.error(f'Error while computing SHAP values: {e}. Using shap_values method instead.')
            shap_values = explainer.shap_values(data)
        self.shap_values = shap_values
        return shap_values

    def beeswarm_plot(self, path: str = None, **kwargs) -> None:
        """
        Plot the SHAP values of all the features as a beeswarm plot.

        Parameters
        ----------
        path: str
            Path to save the plot to.
        kwargs:
            Additional keyword arguments for the plot function:
            see:
            https://github.com/slundberg/shap/blob/45b85c1837283fdaeed7440ec6365a886af4a333/shap/plots/_beeswarm.py#L23
        """
        if path:
            plt.figure()
            shap.plots.beeswarm(self.shap_values, show=False, **kwargs)
            plt.gcf().set_size_inches(10, 6)
            plt.tight_layout()
            plt.savefig(path)
        else:
            shap.plots.beeswarm(self.shap_values, **kwargs)

    def bar_plot(self, path: str = None, **kwargs) -> None:
        """
        Plot the SHAP values of all the features as a beeswarm plot.

        Parameters
        ----------
        path: str
            Path to save the plot to.
        kwargs:
            Additional keyword arguments for the plot function:
            see: https://github.com/slundberg/shap/blob/45b85c1837283fdaeed7440ec6365a886af4a333/shap/plots/_bar.py#L19
        """
        if path:
            plt.figure()
            shap.plots.bar(self.shap_values, show=False, **kwargs)
            plt.gcf().set_size_inches(10, 6)
            plt.tight_layout()
            plt.savefig(path)
        else:
            shap.plots.bar(self.shap_values, **kwargs)

    def sample_explanation_plot(self, index: int, plot_type: str = 'waterfall', path: str = None, **kwargs) -> None:
        """
        Plot the SHAP values of a single sample.

        Parameters
        ----------
        index: int
            Index of the sample to explain
        plot_type: str
            Type of plot to use. Can be 'waterfall' or 'force'
        path: str
            Path to save the plot to.
        kwargs:
            Additional arguments for the plot function.
            see:https://github.com/slundberg/shap/blob/45b85c1837283fdaeed7440ec6365a886af4a333/shap/plots/_force.py#L33
            https://github.com/slundberg/shap/blob/45b85c1837283fdaeed7440ec6365a886af4a333/shap/plots/_waterfall.py#L18
        """
        if plot_type == 'waterfall':
            # visualize the nth prediction's explanation
            if path:
                plt.figure()
                shap.plots.waterfall(self.shap_values[index], show=False, **kwargs)
                plt.gcf().set_size_inches(10, 6)
                plt.tight_layout()
                plt.savefig(path)
            else:
                shap.plots.waterfall(self.shap_values[index], **kwargs)
        elif plot_type == 'force':
            shap.initjs()
            if path:
                new_file_path = str(Path(path).with_suffix('.html'))
                plot = shap.plots.force(self.shap_values[index], show=False, **kwargs)
                shap.save_html(new_file_path, plot)
            else:
                shap.plots.force(self.shap_values[index], **kwargs)
            shap.initjs()
        else:
            raise ValueError('Plot type must be waterfall or force!')

    def feature_explanation_plot(self, index: int, path: str = None, **kwargs) -> None:
        """
        Plot the SHAP values of a single feature.

        Parameters
        ----------
        index: int
            Index of the feature to explain
        path: str
            Path to save the plot to.
        kwargs:
            Additional arguments for the plot function.
            see:
            https://github.com/slundberg/shap/blob/45b85c1837283fdaeed7440ec6365a886af4a333/shap/plots/_scatter.py#L19
        """
        # create a dependence scatter plot to show the effect of a single feature across the whole dataset
        if path:
            plt.figure()
            shap.plots.scatter(self.shap_values[:, index], color=self.shap_values[:, index], show=False, **kwargs)
            plt.gcf().set_size_inches(10, 6)
            plt.tight_layout()
            plt.savefig(path)
        else:
            shap.plots.scatter(self.shap_values[:, index], color=self.shap_values[:, index], **kwargs)

    def heatmap_plot(self, path: str = None, **kwargs) -> None:
        """
        Plot the SHAP values of all the features as a heatmap.

        Parameters
        ----------
        path: str
            Path to save the plot to.
        kwargs:
            Additional arguments for the plot function.
            see:
            https://github.com/slundberg/shap/blob/45b85c1837283fdaeed7440ec6365a886af4a333/shap/plots/_heatmap.py#L12
        """
        if path:
            plt.figure()
            shap.plots.heatmap(self.shap_values, show=False, **kwargs)
            plt.gcf().set_size_inches(10, 6)
            plt.tight_layout()
            plt.savefig(path)
        else:
            shap.plots.heatmap(self.shap_values, **kwargs)

    def positive_class_plot(self, path: str = None, **kwargs) -> None:
        """
        Plot the SHAP values of the positive class as a bar plot.

        Parameters
        ----------
        path: str
            Path to save the plot to.
        kwargs:
            Additional arguments for the plot function.
            see: https://github.com/slundberg/shap/blob/45b85c1837283fdaeed7440ec6365a886af4a333/shap/plots/_bar.py#L19
        """
        if self.mode != 'classification':
            raise ValueError('This method is only available for binary classification models.')
        shap_values2 = self.shap_values[..., 1]
        if path:
            plt.figure()
            shap.plots.bar(shap_values2, show=False, **kwargs)
            plt.gcf().set_size_inches(10, 6)
            plt.tight_layout()
            plt.savefig(path)
        else:
            shap.plots.bar(shap_values2, **kwargs)

    def negative_class_plot(self, path: str = None, **kwargs) -> None:
        """
        Plot the SHAP values of the positive class as a bar plot.

        Parameters
        ----------
        path: str
            Path to save the plot to.
        kwargs:
            Additional arguments for the plot function.
            see: https://github.com/slundberg/shap/blob/45b85c1837283fdaeed7440ec6365a886af4a333/shap/plots/_bar.py#L19
        """
        if self.mode != 'classification':
            raise ValueError('This method is only available for binary classification models.')
        shap_values2 = self.shap_values[..., 0]
        if path:
            plt.figure()
            shap.plots.bar(shap_values2, show=False, **kwargs)
            plt.gcf().set_size_inches(10, 6)
            plt.tight_layout()
            plt.savefig(path)
        else:
            shap.plots.bar(shap_values2, **kwargs)

    def decision_plot(self, path: str = None, **kwargs) -> None:
        """
        Plot the SHAP values of all the features as a decision plot.

        Parameters
        ----------
        path: str
            Path to save the plot to.
        kwargs:
            Additional arguments for the plot function.
            see:
            https://github.com/slundberg/shap/blob/45b85c1837283fdaeed7440ec6365a886af4a333/shap/plots/_decision.py#L222
        """
        # check if the explainer has an expected value
        if not hasattr(self.explainer, 'expected_value'):
            raise ValueError('The explainer does not support an expected value.')
        expected_value = self.explainer.expected_value
        if path:
            plt.figure()
            shap.plots.decision(self.shap_values, show=False, **kwargs)
            plt.gcf().set_size_inches(10, 6)
            plt.tight_layout()
            plt.savefig(path)
        else:
            shap.plots.decision(expected_value, self.shap_values, **kwargs)

