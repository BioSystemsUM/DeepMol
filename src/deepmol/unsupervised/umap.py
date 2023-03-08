import numpy as np
import umap

from deepmol.datasets import Dataset, SmilesDataset
from deepmol.unsupervised.base_unsupervised import UnsupervisedLearn
import plotly.express as px


class UMAP(UnsupervisedLearn):
    """
    Class to perform Uniform Manifold Approximation and Projection (UMAP).

    Wrapper around umap package.
    (https://github.com/lmcinnes/umap)
    """

    def __init__(self,
                 parametric: bool = False,
                 n_neighbors: int = 15,
                 n_components: int = 2,
                 metric: str = 'euclidean',
                 n_epochs: int = None,
                 learning_rate: float = 1.0,
                 low_memory: bool = True,
                 random_state: int = None,
                 **kwargs):
        """
        Initialize UMAP.

        Parameters
        ----------
        parametric : bool
            If True, use parametric UMAP.
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
        kwargs:
            Additional keyword arguments for the UMAP class.
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.low_memory = low_memory
        self.random_state = random_state
        if parametric:
            self.umap = umap.parametric_umap.ParametricUMAP(n_neighbors=self.n_neighbors,
                                                            n_components=self.n_components,
                                                            metric=self.metric,
                                                            n_epochs=self.n_epochs,
                                                            learning_rate=self.learning_rate,
                                                            low_memory=self.low_memory,
                                                            random_state=self.random_state,
                                                            **kwargs)
        else:
            self.umap = umap.UMAP(n_neighbors=self.n_neighbors,
                                  n_components=self.n_components,
                                  metric=self.metric,
                                  n_epochs=self.n_epochs,
                                  learning_rate=self.learning_rate,
                                  low_memory=self.low_memory,
                                  random_state=self.random_state,
                                  **kwargs)

    def _run_unsupervised(self, dataset: Dataset, **kwargs) -> SmilesDataset:
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
        SmilesDataset
            The dataset with the new features.
        """
        self.dataset = dataset
        x_new = self.umap.fit_transform(dataset.X)
        feature_names = [f'UMAP_{i}' for i in range(x_new.shape[1])]
        return SmilesDataset(smiles=dataset.smiles,
                             mols=dataset.mols,
                             X=x_new,
                             y=dataset.y,
                             ids=dataset.ids,
                             feature_names=feature_names,
                             label_names=dataset.label_names,
                             mode=dataset.mode)

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
        self.logger.info(f'{self.n_components} Components UMAP: ')

        if self.dataset.mode == 'classification':
            y = [str(i) for i in self.dataset.y]
        else:
            y = self.dataset.y

        if x_new.shape[1] == 2:
            fig = px.scatter(x_new, x=0, y=1, color=y,
                             labels={'0': 'PC 1', '1': 'PC 2', 'color': self.dataset.label_names[0]}, **kwargs)
        elif x_new.shape[1] == 3:
            fig = px.scatter_3d(x_new, x=0, y=1, z=2, color=y,
                                labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3', 'color': self.dataset.label_names[0]})
        else:
            labels = {str(i): f"UMAP {i + 1}" for i in range(self.n_components)}
            labels['color'] = self.dataset.label_names[0]
            fig = px.scatter_matrix(x_new,
                                    color=y,
                                    dimensions=range(self.n_components),
                                    labels=labels,
                                    **kwargs)
            fig.update_traces(diagonal_visible=False)
        fig.show()
        if path is not None:
            fig.write_image(path)
