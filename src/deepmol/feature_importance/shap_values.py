import pandas as pd
import shap
from matplotlib import pyplot as plt

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
            - 'permutation': Permutation explainer
            - 'exact': Exact explainer
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
            - 'impute': Impute masker
            - 'text': Text masker
        """
        self.explainer = explainer
        self.masker = masker
        self.shap_values = None

    def compute_shap(self, dataset: Dataset, model: Model, **kwargs):
        data = pd.DataFrame(dataset.X, columns=dataset.feature_names, dtype=float)
        kwargs = kwargs
        if self.masker is not None:
            masker_kwargs = masker_args(self.masker, **kwargs)
            masker = str_to_masker(self.masker)(data, **masker_kwargs)
            [kwargs.pop(k) for k in masker_kwargs.keys() if k in kwargs]
            explainer = str_to_explainer(self.explainer)(model.model.predict, masker=masker)
        else:
            explainer = str_to_explainer(self.explainer)(model.model.predict, data)

        self.shap_values = explainer(data, **kwargs)
        return self.shap_values

    def plot_sample_explanation(self, index: int = 0, plot_type: str = 'waterfall', **kwargs):
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
        if self.shap_values is None:
            print('Shap values not computed yet! Computing shap values...')
            self.computeShap(plot=False)

        if plot_type == 'waterfall':
            # visualize the nth prediction's explanation
            shap.plots.waterfall(self.shap_values[index], **kwargs)
        elif plot_type == 'force':
            shap.initjs()
            # visualize the first prediction's explanation with a force plot
            shap.plots.force(self.shap_values[index], **kwargs)
        else:
            raise ValueError('Plot type must be waterfall or force!')

    def plot_feature_explanation(self, index: int = None, **kwargs):
        """
        Plot the SHAP values of a single feature.

        Parameters
        ----------
        index: int
            Index of the feature to explain
        kwargs:
            Additional arguments for the plot function.
        """
        if index is None:
            # summarize the effects of all the features
            shap.plots.beeswarm(self.shap_values, **kwargs)
        else:
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
        if self.shap_values is not None:
            shap.plots.heatmap(self.shap_values, **kwargs)
        else:
            raise ValueError('Shap values not computed yet!')

    # TODO: check this again
    '''
    def plotPositiveClass(self):
        shap_values2 = self.shap_values[...,1]
        print(shap_values2)
        shap.plots.bar(shap_values2)

    def plotNegativeClass(self):
        shap_values2 = self.shap_values[...,0]
        shap.plots.bar(shap_values2)
    '''
