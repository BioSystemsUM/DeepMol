from typing import Tuple

import numpy as np
import umap

from deepmol.datasets import Dataset
from deepmol.unsupervised.base_unsupervised import UnsupervisedLearn
import plotly.express as px


class UMAP(UnsupervisedLearn):
    """
    Class to perform Uniform Manifold Approximation and Projection (UMAP).

    Wrapper around umap package.
    (https://github.com/lmcinnes/umap)
    """

    def __init__(self, parametric: bool = True, **kwargs):
        """
        Initialize UMAP.

        Parameters
        ----------
        parametric : bool
                If True, use parametric UMAP.
        kwargs:
            Additional keyword arguments for the UMAP class (see https://github.com/lmcinnes/umap). Includes:
            n_neighbors : int
                The size of local neighborhood.
            n_components : int
                The dimension of the space to embed into.
            metric : str
                The metric to use for the computation.
            n_epochs : int
                The number of training epochs to use when optimizing the low dimensional embedding.
            learning_rate : float
                The initial learning rate for the embedding optimization.
            low_memory : bool
                If True, use a more memory efficient nearest neighbor implementation.
            random_state : int
                The random seed to use.
        """
        super().__init__()
        if parametric:
            self.umap = umap.parametric_umap.ParametricUMAP(**kwargs)
        else:
            self.umap = umap.UMAP(**kwargs)

    def _run_unsupervised(self, dataset: Dataset, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        dataset : Dataset
            The dataset to run the unsupervised learning on.
        kwargs:
            Additional keyword arguments for the UMAP class.

        Returns
        -------
        x_new : np.ndarray
            The new features.
        feature_names : np.ndarray
            The names of the new features.
        """
        self.dataset = dataset
        x_new = self.umap.fit_transform(dataset.X)
        feature_names = np.array([f'UMAP_{i}' for i in range(x_new.shape[1])])
        return x_new, feature_names

    def plot(self, x_new: np.ndarray, path: str = None, **kwargs) -> None:
        """
        Plot the UMAP embedding.

        Parameters
        ----------
        x_new : np.ndarray
            The new features.
        path : str
            The path to save the plot.
        kwargs:
            Additional keyword arguments for the plot.
        """
        self.logger.info(f'{x_new.shape[1]} Components UMAP: ')

        if self.dataset.mode == 'classification':
            y = [str(i) for i in self.dataset.y]
        else:
            y = self.dataset.y

        if x_new.shape[1] == 2:
            fig = px.scatter(x_new, x=0, y=1, color=y,
                             labels={'0': 'UMAP 1', '1': 'UMAP 2', 'color': self.dataset.label_names[0]}, **kwargs)
        elif x_new.shape[1] == 3:
            fig = px.scatter_3d(x_new, x=0, y=1, z=2, color=y,
                                labels={'0': 'UMAP 1', '1': 'UMAP 2', '2': 'UMAP 3', 'color': self.dataset.label_names[0]})
        else:
            labels = {str(i): f"UMAP {i + 1}" for i in range(x_new.shape[1])}
            labels['color'] = self.dataset.label_names[0]
            fig = px.scatter_matrix(x_new,
                                    color=y,
                                    dimensions=range(x_new.shape[1]),
                                    labels=labels,
                                    **kwargs)
            fig.update_traces(diagonal_visible=False)
        fig.show()
        if path is not None:
            fig.write_image(path)

    def _fit(self, dataset: Dataset) -> 'UMAP':
        """
        Fit the model with dataset.X.

        Parameters
        ----------
        dataset: Dataset
            The dataset to perform unsupervised learning.

        Returns
        -------
        self: TSNE
            The fitted model.
        """
        self.dataset = dataset
        self.umap.fit(dataset.X)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Apply dimensionality reduction on dataset.X.

        Parameters
        ----------
        dataset: Dataset
            The dataset to perform unsupervised learning.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.
        """
        dataset._X = self.umap.transform(dataset.X)
        dataset.feature_names = np.array([f'UMAP_{i}' for i in range(dataset.X.shape[1])])
        return dataset
