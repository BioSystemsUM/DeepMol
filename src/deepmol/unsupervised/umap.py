from deepmol.unsupervised.base_unsupervised import UnsupervisedLearn
from umap.parametric_umap import ParametricUMAP
import plotly.express as px


class UMAP(UnsupervisedLearn):
    """
    Class to perform Uniform Manifold Approximation and Projection (UMAP).

    Wrapper around umap package.
    (https://github.com/lmcinnes/umap)
    """

    def __init__(self,
                 n_neighbors: int = 15,
                 n_components: int = 2,
                 metric: str = 'euclidean',
                 n_epochs: int = None,
                 learning_rate: float = 1.0,
                 low_memory: bool = True,
                 random_state: int = None):
        """
        Initialize UMAP.

        Parameters
        ----------
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
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.low_memory = low_memory
        self.random_state = random_state

    def _runUnsupervised(self, plot=True):
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        plot : bool
            If True, plot the embedding.

        Returns
        -------
        embedding : array
            The embedding of the training data in low-dimensional space.
        """

        embedder = ParametricUMAP(n_neighbors=self.n_neighbors,
                                  n_components=self.n_components,
                                  metric=self.metric,
                                  n_epochs=self.n_epochs,
                                  learning_rate=self.learning_rate,
                                  low_memory=self.low_memory,
                                  random_state=self.random_state)

        embedding = embedder.fit_transform(self.features)

        if plot:
            if self.n_components != 2:
                raise ValueError('Only 2 components UMAP supported!')

            self._plot(embedding, self.dataset.y)
            # points(embedding, labels=self.dataset.y)

        return embedding

    def _plot(self, embedding, Y_train):
        """
        Plot the embedding.

        Parameters
        ----------
        embedding : array
            The embedding of the training data in low-dimensional space.
        Y_train : array
            The labels of the training data.
        """
        print('2 Components UMAP: ')

        dic = {0: "Not Active (0)", 1: "Active (1)"}
        colors_map = []
        for elem in self.dataset.y:
            colors_map.append(dic[elem])

        fig = px.scatter(embedding, x=0, y=1,
                         color=colors_map, labels={'color': 'Class'},
                         title='UMAP:'
                         )
        fig.show()
