import pandas as pd
import shap

from deepmol.datasets import Dataset
from deepmol.feature_importance._utils import str_to_explainer, str_to_masker, masker_args
from deepmol.models.models import Model


class ShapValues:
    """
    SHAP (SHapley Additive exPlanations) wrapper for DeepMol
    It allows to compute and analyze the SHAP values of DeepMol models.
    """

    def __init__(self, explainer: str = 'permutation', masker: str = None):
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
            - 'tree': Tree explainer
            - 'gpu_tree': GPU Tree explainer
            - 'partition': Partition explainer
            - 'linear': Linear explainer
            - 'sampling': Sampling explainer
            - 'deep': Deep explainer
        masker: str
            The masker to use. It can be one of the following:
            - 'independent': Independent masker
            - 'partition': Partition masker
        """
        self.explainer = explainer
        self.masker = masker
        self.shap_values = None

    def compute_shap(self, dataset: Dataset, model: Model, **kwargs):
        data = pd.DataFrame(dataset.X, columns=dataset.feature_names, dtype=float)
        kwargs = kwargs
        if self.explainer == 'gpu_tree':
            raise ValueError('Tree explainer not supported yet! Referring to '
                             'https://github.com/slundberg/shap/issues/1136 and '
                             'https://github.com/slundberg/shap/issues/1650')
        #elif self.explainer in ['deep', 'linear', 'gradient', 'tree', 'gpu_tree']:
        #    model_instance = model.model
        elif self.explainer in ['tree', 'linear', 'gradient', 'deep']:
            model_instance = model.model
        else:
            if dataset.mode == 'classification':
                model_instance = model.model.predict_proba
            else:
                model_instance = model.model.predict
        if self.masker is not None:
            masker_kwargs = masker_args(self.masker, **kwargs)
            if self.masker == 'text':
                masker = str_to_masker(self.masker)(**masker_kwargs)
            else:
                masker = str_to_masker(self.masker)(data, **masker_kwargs)
            [kwargs.pop(k) for k in masker_kwargs.keys() if k in kwargs]
            if self.explainer == 'sampling':
                explainer = str_to_explainer(self.explainer)(model_instance, data, masker=masker, **kwargs)
            else:
                explainer = str_to_explainer(self.explainer)(model_instance, masker=masker)
        else:
            explainer = str_to_explainer(self.explainer)(model_instance, data)

        self.shap_values = explainer(data, **kwargs)
        return self.shap_values

    def beeswarm_plot(self, **kwargs):
        shap.plots.beeswarm(self.shap_values, **kwargs)

    def bar_plot(self, **kwargs):
        """
        Plot the SHAP values of all the features as a bar plot.

        Parameters
        ----------
        kwargs:
            Additional keyword arguments for the plot function:
            max_display: int
                Maximum number of features to display.
            order: str
                Ordered features. By default, the features are ordered by the absolute value of the SHAP value.
            clustering:
            clustering_cutoff=0.5
            merge_cohorts=False
            show_data="auto"
            show=True

        """
        shap.plots.bar(self.shap_values, **kwargs)

    def sample_explanation_plot(self, index: int, plot_type: str = 'waterfall', **kwargs):
        """
        Plot the SHAP values of a single sample.

        Parameters
        ----------
        index: int
            Index of the sample to explain
        plot_type: str
            Type of plot to use. Can be 'waterfall' or 'force'
        kwargs:
            Additional arguments for the plot function.
        """
        if plot_type == 'waterfall':
            # visualize the nth prediction's explanation
            shap.plots.waterfall(self.shap_values[index], **kwargs)
        elif plot_type == 'force':
            shap.initjs()
            # visualize the first prediction's explanation with a force plot
            shap.plots.force(self.shap_values[index], **kwargs)
        else:
            raise ValueError('Plot type must be waterfall or force!')

    def feature_explanation_plot(self, index: int, **kwargs):
        """
        Plot the SHAP values of a single feature.

        Parameters
        ----------
        index: int
            Index of the feature to explain
        kwargs:
            Additional arguments for the plot function.
        """
        # create a dependence scatter plot to show the effect of a single feature across the whole dataset
        shap.plots.scatter(self.shap_values[:, index], color=self.shap_values[:, index], **kwargs)

    def plot_heat_map(self, **kwargs):
        """
        Plot the SHAP values of all the features as a heatmap.

        Parameters
        ----------
        kwargs:
            Additional arguments for the plot function.
        """
        shap.plots.heatmap(self.shap_values, **kwargs)

    def positive_class_plot(self):
        shap_values2 = self.shap_values[..., 1]
        print(shap_values2)
        shap.plots.bar(shap_values2)

    def negative_class_plot(self):
        shap_values2 = self.shap_values[..., 0]
        shap.plots.bar(shap_values2)

